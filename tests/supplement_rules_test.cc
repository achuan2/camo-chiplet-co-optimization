#ifdef NDEBUG
#undef NDEBUG
#endif
#include "moham.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
bool gTerminateEval = false;

namespace moham {
struct SupplementTestAccess {
    static void Run() {
        Graph graph;
        std::vector<ArchTemplate> templates;
        Config config;
        config.seed = 123;
        config.moham.algorithm = "c-taea";
        config.moham.objective_mode = "slo-risk";
        config.moham.thermal_enabled = true;
        config.moham.population_size = 4;
        Moham optimizer(config, &graph, &templates, {});
        using I = Moham::Individual;
        auto make = [](double x, double y, double cv = 0.0) {
            I a; a.objectives = {{x,y,1,1,1}}; a.constraint_violation = cv; return a;
        };
        optimizer.ctaea_convergence_archive_ = {make(2,2),make(0,4)};
        optimizer.ctaea_diversity_archive_ = {make(1,1),make(4,0)};
        optimizer.UpdateCTAEAMatingProbabilities();
        assert(optimizer.ctaea_first_parent_from_convergence_ == 0.0);
        assert(optimizer.ctaea_second_parent_convergence_probability_ == 1.0);
        optimizer.config_.moham.ctaea_second_parent_rule = "joint-contribution";
        optimizer.UpdateCTAEAMatingProbabilities();
        assert(optimizer.ctaea_second_parent_convergence_probability_ == 0.25);
        optimizer.ctaea_convergence_archive_ = {make(1,1)};
        optimizer.ctaea_diversity_archive_ = {make(1,1)};
        optimizer.UpdateCTAEAMatingProbabilities();
        assert(optimizer.ctaea_second_parent_convergence_probability_ == 0.5);
        assert(optimizer.ctaea_first_parent_from_convergence_ == 0.0);
        optimizer.ctaea_convergence_archive_.clear();
        optimizer.UpdateCTAEAMatingProbabilities();
        assert(optimizer.ctaea_second_parent_convergence_probability_ == 0.0);
        assert(&optimizer.SelectCTAEAParent(true) == &optimizer.ctaea_diversity_archive_[0]);
        // Both archives empty is not a supported mating precondition.
        assert(!optimizer.ctaea_diversity_archive_.empty());
        Moham::Population archive = {make(1,1,1),make(2,2,1),make(3,3,1),make(4,4,1)};
        bool redraw_outside_pair = false;
        for (unsigned seed = 1; seed <= 200; ++seed) {
            std::mt19937_64 mirror(seed);
            const auto first = mirror() % archive.size();
            const auto second = mirror() % archive.size();
            const auto third = mirror();
            optimizer.rng_[0].seed(seed);
            optimizer.config_.moham.ctaea_infeasible_tournament_rule = "candidate-pair";
            const auto& pair_result = optimizer.CTAEAMatingSelection(archive);
            assert(&pair_result == &archive[first] || &pair_result == &archive[second]);
            assert(&pair_result == &archive[third % 2 ? second : first]);
            optimizer.rng_[0].seed(seed);
            optimizer.config_.moham.ctaea_infeasible_tournament_rule = "archive-redraw";
            const auto& redraw = optimizer.CTAEAMatingSelection(archive);
            assert(&redraw == &archive[third % archive.size()]);
            redraw_outside_pair |= &redraw != &archive[first] && &redraw != &archive[second];
        }
        assert(redraw_outside_pair);
        for (unsigned seed = 1; seed <= 100; ++seed) {
            optimizer.rng_[0].seed(seed);
            auto mixed = Moham::Population{make(5,5,0),make(0,0,1)};
            std::mt19937_64 mixed_rng(seed);
            const auto m1 = mixed_rng() % 2;
            const auto m2 = mixed_rng() % 2;
            const auto& mixed_result = optimizer.CTAEAMatingSelection(mixed);
            if (m1 != m2) assert(mixed_result.constraint_violation == 0.0);
            optimizer.rng_[0].seed(seed);
            auto feasible = Moham::Population{make(0,0),make(1,1)};
            std::mt19937_64 mirror(seed);
            auto a = mirror() % 2; auto b = mirror() % 2;
            const auto& selected = optimizer.CTAEAMatingSelection(feasible);
            assert(&selected == &feasible[a == 0 || b == 0 ? 0 : 1]);
        }
        optimizer.optimizer_start_ = std::chrono::steady_clock::now();
        optimizer.wall_clock_limit_active_ = true;
        optimizer.wall_clock_deadline_ = optimizer.optimizer_start_;
        for (int context = 0; context < 3; ++context) {
            I candidate; candidate.valid = true;
            assert(!optimizer.Evaluate(candidate, context == 2));
            assert(!candidate.valid);
        }
        assert(optimizer.evaluation_count_.load() == 0);
        assert(optimizer.repair_evaluation_count_.load() == 0);
        assert(optimizer.evaluation_records_.empty());
        std::cout << "PASS: probability, duplicates, fallback, redraw support, tournament and admission rejection\n";
    }
};
}
int main() { moham::SupplementTestAccess::Run(); }
