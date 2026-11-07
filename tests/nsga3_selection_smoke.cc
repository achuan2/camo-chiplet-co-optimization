#include "nsga.h"

#include <cassert>
#include <random>
#include <vector>

// Header-only smoke test for the two invariants SCBA-NSGA-III relies on:
// feasibility-first ranking and five-objective NSGA-III environmental selection.
class Harness : public moham::NSGA
{
public:
    struct Candidate : Individual<5> {};
    using Population = std::vector<Candidate>;

    void Rank(Population& population)
    {
        ConfigureNSGAIII(true, 2, true);
        ConfigureObjectiveCount(5);
        ConfigureReferenceDirectionBias({1.0, 1.2, 1.0, 1.3, 1.5});
        AssignRankAndCrowdingDistance(population);
    }

    void Select(Population& destination, const Population& source)
    {
        std::mt19937_64 rng(7);
        Survival(destination, source, false, rng);
    }

    void RankConstrainedNSGAII(Population& population)
    {
        ConfigureNSGAIII(false, 0, true);
        ConfigureObjectiveCount(5);
        AssignRankAndCrowdingDistance(population);
    }
};

int main()
{
    Harness::Population population(6);
    for (std::size_t i = 0; i < population.size(); ++i)
    {
        population[i].valid = true;
        population[i].objectives = {{static_cast<double>(i), static_cast<double>(5 - i), 1.0, 2.0, 1.0 + i}};
    }

    // A better-but-infeasible solution must not dominate a feasible solution.
    population[0].constraint_violation = 0.0;
    population[1].constraint_violation = 0.5;
    population[1].objectives = {{0.0, 0.0, 0.0, 0.0, 0.0}};

    Harness harness;
    harness.Rank(population);
    assert(population[0].rank < population[1].rank);

    auto nsga2_population = population;
    Harness constrained_nsga2;
    constrained_nsga2.RankConstrainedNSGAII(nsga2_population);
    assert(nsga2_population[0].rank < nsga2_population[1].rank);

    Harness::Population survivors(3);
    harness.Select(survivors, population);
    assert(survivors.size() == 3);
    for (const auto& survivor : survivors)
        assert(survivor.constraint_violation == 0.0);
}
