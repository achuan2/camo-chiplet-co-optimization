#ifndef MOHAM_NSGA_H_
#define MOHAM_NSGA_H_

#include <random>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <cmath>
#include <limits>
#include <cstdint>
#include <utility>

#define ENERGY_OBJ 0
#define LATENCY_OBJ 1
#define AREA_OBJ 2

namespace moham 
{
    enum Dominance
    {
        DOMINATING,
        DOMINATED,
        FRONTIER
    };

    class NSGA 
    {
    protected:

        template <std::size_t N>
        struct Individual {
            unsigned rank = 0;
            double crowding_distance = 0.0;
            std::array<double, N> objectives{};
            // Constraint violation is the sum of normalised positive violations.
            // It is zero for all legacy (unconstrained) searches.
            double constraint_violation = 0.0;
            bool valid = false;
        };

        // https://hal.inria.fr/hal-01909120/document
        std::vector<double> gen_stability_measures_;
        double stability_;

        bool single_obj_ = false, prod_obj_;
        std::vector<double> weights_obj_; // energy, latency, area
        bool constraint_domination_ = false;
        bool nsga3_selection_ = false;
        unsigned reference_partitions_ = 0;
        // The top-level individual reserves five slots so comparison runs can
        // share one encoding. Legacy MOHaM can still rank only E/Cmax/A.
        unsigned active_objectives_ = 0;
        // SCBA-NSGA-III can reweight (but never remove) simplex directions
        // according to feasible-space coverage and constraint-boundary pressure.
        std::vector<double> reference_direction_bias_;

        void ConfigureNSGAIII(bool enabled, unsigned reference_partitions, bool constrained)
        {
            nsga3_selection_ = enabled;
            // Constraint-domination is also used by the fair c-nsga2
            // baseline; only environmental niching is NSGA-III-specific.
            constraint_domination_ = constrained;
            reference_partitions_ = reference_partitions;
        }

        void ConfigureReferenceDirectionBias(std::vector<double> bias)
        {
            reference_direction_bias_ = std::move(bias);
        }

        void ConfigureObjectiveCount(unsigned count)
        {
            active_objectives_ = count;
        }

        template <std::size_t N>
        std::size_t ObjectiveCount() const
        {
            return active_objectives_ == 0 ? N : std::min<std::size_t>(active_objectives_, N);
        }

    protected:

        template <typename P>
        void Merging(P& merged_pop, const P& parent_pop, const P& offspring_pop, const P& immigrant_pop)
        {
            merged_pop.resize(parent_pop.size() + immigrant_pop.size());
            merged_pop.reserve(parent_pop.size() + immigrant_pop.size() + offspring_pop.size());
            
            std::copy(
                parent_pop.begin(),
                parent_pop.end(),
                merged_pop.begin());
            std::copy(
                immigrant_pop.begin(),
                immigrant_pop.end(),
                merged_pop.begin() + parent_pop.size());
            std::copy_if(
                offspring_pop.begin(),
                offspring_pop.end(),
                std::back_inserter(merged_pop),
                [](auto& ind) { return ind.valid; });
        }

        static double ReferenceDirectionCount(std::size_t dimensions, unsigned partitions)
        {
            // Number of simplex-lattice directions: C(H + M - 1, M - 1).
            double result = 1.0;
            for (std::size_t i = 1; i < dimensions; ++i)
                result = result * (partitions + i) / i;
            return result;
        }

        static void BuildReferenceDirectionsRecursive(
            std::size_t dimensions,
            unsigned partitions,
            std::size_t dimension,
            unsigned remaining,
            std::vector<double>& current,
            std::vector<std::vector<double>>& directions)
        {
            if (dimension + 1 == dimensions)
            {
                current[dimension] = static_cast<double>(remaining) / partitions;
                directions.push_back(current);
                return;
            }

            for (unsigned value = 0; value <= remaining; ++value)
            {
                current[dimension] = static_cast<double>(value) / partitions;
                BuildReferenceDirectionsRecursive(dimensions, partitions, dimension + 1,
                                                  remaining - value, current, directions);
            }
        }

        static std::vector<std::vector<double>> BuildReferenceDirections(std::size_t dimensions, unsigned partitions)
        {
            partitions = std::max(1u, partitions);
            std::vector<std::vector<double>> directions;
            std::vector<double> current(dimensions, 0.0);
            BuildReferenceDirectionsRecursive(dimensions, partitions, 0, partitions, current, directions);
            return directions;
        }

        template <typename P, typename URNG>
        void SurvivalNSGAIII(P& new_pop, const P& source_pop, const bool shuffle, URNG& rng)
        {
            using IndividualType = typename P::value_type;
            constexpr std::size_t N = std::tuple_size<decltype(std::declval<IndividualType>().objectives)>::value;
            const std::size_t active_dimensions = ObjectiveCount<N>();
            const std::size_t target_size = new_pop.size();
            assert(source_pop.size() >= target_size);

            unsigned partitions = reference_partitions_;
            if (partitions == 0)
            {
                partitions = 1;
                while (ReferenceDirectionCount(active_dimensions, partitions) < target_size)
                    ++partitions;
            }
            auto directions = BuildReferenceDirections(active_dimensions, partitions);
            if (reference_direction_bias_.size() >= active_dimensions)
            {
                for (auto& direction : directions)
                {
                    double sum = 0.0;
                    for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                    {
                        direction[objective] *= std::max(reference_direction_bias_[objective], 1e-6);
                        sum += direction[objective];
                    }
                    if (sum > 0.0)
                        for (double& value : direction) value /= sum;
                }
            }

            P selected;
            selected.reserve(target_size);
            unsigned splitting_rank = 0;
            while (true)
            {
                std::vector<const IndividualType*> front;
                for (const auto& individual : source_pop)
                    if (individual.rank == splitting_rank)
                        front.push_back(&individual);
                if (front.empty()) break;
                if (selected.size() + front.size() > target_size) break;
                for (const auto* individual : front)
                    selected.push_back(*individual);
                ++splitting_rank;
            }
            if (selected.size() == target_size)
            {
                new_pop = std::move(selected);
                if (shuffle) std::shuffle(new_pop.begin(), new_pop.end(), rng);
                return;
            }

            // Deb and Jain's NSGA-III normalization: translate by the ideal
            // point, identify one ASF extreme point per objective, and use the
            // resulting hyperplane intercepts. Degenerate/singular fronts use
            // the component-wise worst point, as prescribed by the fallback.
            std::array<double, N> min_objectives;
            std::array<double, N> ranges;
            min_objectives.fill(std::numeric_limits<double>::max());
            std::array<double, N> worst_objectives;
            worst_objectives.fill(std::numeric_limits<double>::lowest());
            for (const auto& individual : source_pop)
            {
                if (individual.rank > splitting_rank) continue;
                for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                {
                    min_objectives[objective] = std::min(min_objectives[objective], individual.objectives[objective]);
                    worst_objectives[objective] = std::max(worst_objectives[objective], individual.objectives[objective]);
                }
            }
            for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                ranges[objective] = std::max(worst_objectives[objective] - min_objectives[objective], 1e-12);

            std::vector<std::vector<double>> extreme_points(
                active_dimensions, std::vector<double>(active_dimensions, 0.0));
            for (std::size_t axis = 0; axis < active_dimensions; ++axis)
            {
                const IndividualType* extreme = nullptr;
                double best_asf = std::numeric_limits<double>::max();
                for (const auto& individual : source_pop)
                {
                    if (individual.rank > splitting_rank) continue;
                    double asf = std::numeric_limits<double>::lowest();
                    for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                    {
                        const double translated =
                            std::max(0.0, individual.objectives[objective] - min_objectives[objective]);
                        const double weight = objective == axis ? 1.0 : 1e-6;
                        asf = std::max(asf, translated / weight);
                    }
                    if (asf < best_asf)
                    {
                        best_asf = asf;
                        extreme = &individual;
                    }
                }
                if (extreme)
                    for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                        extreme_points[axis][objective] =
                            std::max(0.0, extreme->objectives[objective] - min_objectives[objective]);
            }

            // Solve A*x=1 with scaled columns and partial pivoting. If
            // y_j=range_j*x_j, then the intercept is range_j/y_j.
            std::vector<std::vector<double>> system(
                active_dimensions, std::vector<double>(active_dimensions + 1, 1.0));
            for (std::size_t row = 0; row < active_dimensions; ++row)
                for (std::size_t column = 0; column < active_dimensions; ++column)
                    system[row][column] = extreme_points[row][column] / ranges[column];

            bool valid_hyperplane = true;
            for (std::size_t column = 0; column < active_dimensions; ++column)
            {
                std::size_t pivot = column;
                for (std::size_t row = column + 1; row < active_dimensions; ++row)
                    if (std::abs(system[row][column]) > std::abs(system[pivot][column]))
                        pivot = row;
                if (std::abs(system[pivot][column]) <= 1e-12)
                {
                    valid_hyperplane = false;
                    break;
                }
                if (pivot != column) std::swap(system[pivot], system[column]);
                const double divisor = system[column][column];
                for (std::size_t entry = column; entry <= active_dimensions; ++entry)
                    system[column][entry] /= divisor;
                for (std::size_t row = 0; row < active_dimensions; ++row)
                {
                    if (row == column) continue;
                    const double factor = system[row][column];
                    for (std::size_t entry = column; entry <= active_dimensions; ++entry)
                        system[row][entry] -= factor * system[column][entry];
                }
            }
            if (valid_hyperplane)
                for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                {
                    const double scaled_solution = system[objective][active_dimensions];
                    const double intercept = scaled_solution > 1e-12
                        ? ranges[objective] / scaled_solution
                        : std::numeric_limits<double>::quiet_NaN();
                    if (!std::isfinite(intercept) || intercept <= 1e-12 ||
                        intercept > ranges[objective] * (1.0 + 1e-6))
                        valid_hyperplane = false;
                    else
                        ranges[objective] = intercept;
                }
            if (!valid_hyperplane)
                for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                    ranges[objective] =
                        std::max(worst_objectives[objective] - min_objectives[objective], 1e-12);

            auto associate = [&](const auto& individual) {
                std::vector<double> normalised(active_dimensions, 0.0);
                for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                    normalised[objective] = (individual.objectives[objective] - min_objectives[objective]) / ranges[objective];

                std::size_t closest = 0;
                double closest_distance = std::numeric_limits<double>::max();
                for (std::size_t direction_id = 0; direction_id < directions.size(); ++direction_id)
                {
                    const auto& direction = directions[direction_id];
                    double direction_norm_sq = 0.0;
                    double projection_numerator = 0.0;
                    double normalised_norm_sq = 0.0;
                    for (std::size_t objective = 0; objective < active_dimensions; ++objective)
                    {
                        direction_norm_sq += direction[objective] * direction[objective];
                        projection_numerator += normalised[objective] * direction[objective];
                        normalised_norm_sq += normalised[objective] * normalised[objective];
                    }
                    const double projection_sq = direction_norm_sq > 0.0
                        ? projection_numerator * projection_numerator / direction_norm_sq : 0.0;
                    const double distance = std::sqrt(std::max(0.0, normalised_norm_sq - projection_sq));
                    if (distance < closest_distance)
                    {
                        closest_distance = distance;
                        closest = direction_id;
                    }
                }
                return std::make_pair(closest, closest_distance);
            };

            std::vector<std::size_t> niche_count(directions.size(), 0);
            for (const auto& individual : selected)
                niche_count[associate(individual).first]++;

            struct Candidate { const IndividualType* individual; std::size_t direction; double distance; };
            std::vector<Candidate> candidates;
            for (const auto& individual : source_pop)
                if (individual.rank == splitting_rank)
                {
                    const auto association = associate(individual);
                    candidates.push_back({&individual, association.first, association.second});
                }

            while (selected.size() < target_size && !candidates.empty())
            {
                std::size_t direction = directions.size();
                std::size_t lowest_niche = std::numeric_limits<std::size_t>::max();
                for (const auto& candidate : candidates)
                    if (niche_count[candidate.direction] < lowest_niche)
                    {
                        lowest_niche = niche_count[candidate.direction];
                        direction = candidate.direction;
                    }

                std::vector<std::size_t> direction_candidates;
                for (std::size_t i = 0; i < candidates.size(); ++i)
                    if (candidates[i].direction == direction)
                        direction_candidates.push_back(i);

                std::size_t chosen = direction_candidates.front();
                if (niche_count[direction] == 0)
                    chosen = *std::min_element(direction_candidates.begin(), direction_candidates.end(),
                        [&](std::size_t a, std::size_t b) { return candidates[a].distance < candidates[b].distance; });
                else
                    chosen = direction_candidates[rng() % direction_candidates.size()];

                selected.push_back(*candidates[chosen].individual);
                niche_count[direction]++;
                candidates.erase(candidates.begin() + chosen);
            }

            // This can only happen for malformed ranks; retain a deterministic fallback.
            if (selected.size() < target_size)
            {
                P fallback = source_pop;
                std::sort(fallback.begin(), fallback.end(), [](const auto& a, const auto& b) {
                    return a.rank < b.rank || (a.rank == b.rank && a.crowding_distance > b.crowding_distance);
                });
                for (const auto& individual : fallback)
                    if (selected.size() < target_size)
                        selected.push_back(individual);
            }

            new_pop = std::move(selected);
            if (shuffle) std::shuffle(new_pop.begin(), new_pop.end(), rng);
        }

        template <typename P, typename URNG>
        void Survival(P& new_pop, const P& source_pop, const bool shuffle, URNG& rng)
        {
            if (nsga3_selection_)
            {
                SurvivalNSGAIII(new_pop, source_pop, shuffle, rng);
                return;
            }
            // Sort by rank and crowding distance and select population_size_
            std::partial_sort_copy(
                source_pop.begin(), source_pop.end(),
                new_pop.begin(), new_pop.end(),
                [&](const auto &a, const auto &b) -> bool
                {
                    return a.rank < b.rank || (a.rank == b.rank && a.crowding_distance > b.crowding_distance);
                });

            // Shuffle
            if (shuffle) std::shuffle(std::begin(new_pop), std::end(new_pop), rng);
        }

        template <std::size_t N>
        double CalculateSingleObj(const Individual<N> &ind) {
            double tot_obj = prod_obj_ ? 1 : 0;
            for (std::size_t i = 0; i < ObjectiveCount<N>(); i++)
            {
                double weight = i < weights_obj_.size() ? weights_obj_[i] : 1;
                if (prod_obj_) {
                    tot_obj *= std::pow(weight, ind.objectives[i]);
                } else {
                    tot_obj += weight * ind.objectives[i];
                }
            }
            return tot_obj;
        }

        template <std::size_t N>
        Dominance CheckDominance(const Individual<N> &a, const Individual<N> &b)
        {
            if (constraint_domination_)
            {
                constexpr double epsilon = 1e-12;
                const bool a_feasible = a.constraint_violation <= epsilon;
                const bool b_feasible = b.constraint_violation <= epsilon;
                if (a_feasible != b_feasible)
                    return a_feasible ? Dominance::DOMINATING : Dominance::DOMINATED;
                if (!a_feasible && !b_feasible)
                {
                    if (a.constraint_violation + epsilon < b.constraint_violation) return Dominance::DOMINATING;
                    if (b.constraint_violation + epsilon < a.constraint_violation) return Dominance::DOMINATED;
                    return Dominance::FRONTIER;
                }
            }
            if (!single_obj_) {
                bool all_a_less_or_equal_than_b = true;
                bool any_a_less_than_b = false;
                bool all_b_less_or_equal_than_a = true;
                bool any_b_less_than_a = false;

                for (std::size_t i = 0; i < ObjectiveCount<N>(); i++)
                {
                    if (a.objectives[i] > b.objectives[i])
                    {
                        all_a_less_or_equal_than_b = false;
                        any_b_less_than_a = true;
                    }
                    else if (b.objectives[i] > a.objectives[i])
                    {
                        any_a_less_than_b = true;
                        all_b_less_or_equal_than_a = false;
                    }
                }

                if (all_a_less_or_equal_than_b && any_a_less_than_b)
                    return Dominance::DOMINATING;
                if (all_b_less_or_equal_than_a && any_b_less_than_a)
                    return Dominance::DOMINATED;
            } 
            else 
            {
                double obj_a = CalculateSingleObj(a);
                double obj_b = CalculateSingleObj(b);

                if (obj_a < obj_b) return Dominance::DOMINATING;
                if (obj_a > obj_b) return Dominance::DOMINATED; 
            }
            return Dominance::FRONTIER;
        }

        template <typename P>
        void AssignCrowdingDistance(P &population, std::vector<uint64_t> &pareto_front)
        {
            for (auto p : pareto_front)
                // population[p].crowding_distance = -std::accumulate(population[p].objectives.begin(), population[p].objectives.end(), 1, std::multiplies<double>());
                population[p].crowding_distance = 0.0;

            if (!single_obj_) {
                constexpr std::size_t N = std::tuple_size<decltype(population[0].objectives)>::value;
                for (std::size_t i = 0; i < ObjectiveCount<N>(); i++)
                {
                    std::sort(pareto_front.begin(), pareto_front.end(), 
                    [&](const uint64_t a, const uint64_t b) -> bool
                        { return population[a].objectives[i] < population[b].objectives[i]; });

                    population[pareto_front.front()].crowding_distance = 10e14;
                    population[pareto_front.back()].crowding_distance = 10e14;

                    double range = population[pareto_front.back()].objectives[i] - population[pareto_front.front()].objectives[i];
                    assert(range >= 0);
                    range = (range == 0.0) ? 1.0 : range;

                    for (uint64_t j = 1; j < pareto_front.size() - 1; j++)
                    {
                        uint64_t r_prev = pareto_front[j - 1];
                        uint64_t r_next = pareto_front[j + 1];
                        uint64_t r_this = pareto_front[j];
                        population[r_this].crowding_distance += std::abs(population[r_next].objectives[i] - population[r_prev].objectives[i]) / range;
                    }
                }
            } else {
                std::sort(pareto_front.begin(), pareto_front.end(), 
                    [&](const uint64_t a, const uint64_t b) -> bool
                        { return CalculateSingleObj(population[a]) < CalculateSingleObj(population[b]); });

                    population[pareto_front.front()].crowding_distance = 10e14;
                    population[pareto_front.back()].crowding_distance = 10e14;

                    double range = CalculateSingleObj(population[pareto_front.back()]) - CalculateSingleObj(population[pareto_front.front()]);
                    assert(range >= 0);
                    range = (range == 0.0) ? 1.0 : range;

                    for (uint64_t j = 1; j < pareto_front.size() - 1; j++)
                    {
                        uint64_t r_prev = pareto_front[j - 1];
                        uint64_t r_next = pareto_front[j + 1];
                        uint64_t r_this = pareto_front[j];
                        population[r_this].crowding_distance += std::abs(CalculateSingleObj(population[r_next]) - CalculateSingleObj(population[r_prev])) / range;
                    }
            }
        }
        
        template <typename P>
        void AssignRankAndCrowdingDistance(P &population)
        {
            std::vector<uint64_t> pareto_front;
            std::vector<std::vector<uint64_t>> dominated_by(population.size(), std::vector<uint64_t>());
            std::vector<uint64_t> num_dominating(population.size(), 0);

            for (uint64_t i = 0; i < population.size(); i++)
            {
                for (uint64_t j = i + 1; j < population.size(); j++)
                {
                    switch (CheckDominance(population[i], population[j]))
                    {
                    case Dominance::DOMINATING:
                        dominated_by[i].push_back(j);
                        num_dominating[j]++;
                        break;
                    case Dominance::DOMINATED:
                        dominated_by[j].push_back(i);
                        num_dominating[i]++;
                        break;
                    case Dominance::FRONTIER:
                        break;
                    }
                }

                if (num_dominating[i] == 0)
                {
                    population[i].rank = 0;
                    pareto_front.push_back(i);
                }
            }

            uint64_t total_debug = 0;
            for (uint64_t f = 0; !pareto_front.empty(); f++)
            {
                total_debug += pareto_front.size();

                AssignCrowdingDistance(population, pareto_front);

                std::vector<uint64_t> new_pareto_front;

                for (uint64_t p : pareto_front)
                {

                    for (uint64_t q : dominated_by[p])
                    {

                        num_dominating[q]--;

                        if (num_dominating[q] == 0)
                        {
                            population[q].rank = f + 1;
                            new_pareto_front.push_back(q);
                        }
                    }
                }
                pareto_front = new_pareto_front;
            }

            assert(total_debug == population.size());
        }

        template <typename P>
        std::string RankString(P& pop) {
            std::stringstream ss;

            unsigned ranked =  0;
            for (unsigned i = 0; i < 4; i++)
            {
                unsigned i_ranked = std::count_if(pop.begin(), pop.end(), [i](const auto& ind){ return ind.rank == i; });
                ss << i << ":" << std::left << std::setw(4) << i_ranked;
                ranked += i_ranked;
            }

            ss << "+:" << std::left << std::setw(4) << pop.size() - ranked;
            
            /*
            for (auto &ind : pop)
            {
                if (ind.rank < 10)
                    ss << ind.rank;
                else if (ind.rank < 36)
                    ss << (char)(ind.rank + 55);
                else if (ind.rank < 61)
                    ss << (char)(ind.rank + 61);
                else
                    ss << "+";
            }
            */
            
            return ss.str();
        }


        template <typename P>
        void UpdateStabilityMeasure(P &pop, unsigned l, std::size_t stability_window) {
            double dl = -1.0;

            for (auto &ind : pop)
                if (ind.crowding_distance > dl)
                    dl = ind.crowding_distance;
            
            gen_stability_measures_[l % stability_window] = dl;
        }


        double CalculateGenerationStability(std::size_t L)
        {
            double dmean = std::accumulate(gen_stability_measures_.begin(), gen_stability_measures_.end(), 0.0) / L;

            return std::sqrt(std::accumulate(
                    gen_stability_measures_.begin(), 
                    gen_stability_measures_.end(),
                    0.0, 
                    [dmean](double psum, double d) {
                        return psum + std::pow(d - dmean, 2);
                    }
                ) / L);
        } 
    };
}

#endif // MOHAM_NSGA_H_
