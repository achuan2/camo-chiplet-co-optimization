#ifndef MOHAM_CONFIG_H_
#define MOHAM_CONFIG_H_

#include <string>
#include <utility>
#include <vector>

namespace moham {

    struct Config 
    {
        struct Medea 
        {
            unsigned num_generations            = 1;
            unsigned population_size            = 100;
            unsigned immigrant_population_size  = 0;
            unsigned num_threads                = 0;
            double fill_mutation_prob           = 0.5;
            double parallel_mutation_prob       = 0.5;
            double random_mutation_prob         = 0.5;
            double crossover_prob               = 0.5;

            bool use_tournament                 = false;
            bool update_ert                     = false;
            bool random_when_illegal            = true;
            unsigned buffer_update_granularity  = 16;

            bool stop_on_convergence            = true;
            unsigned stability_window           = 5;
            double stability_threshold          = 0.04;

        } medea;

        struct Moham 
        {
            unsigned num_generations            = 1;
            unsigned population_size            = 100;
            unsigned immigrant_population_size  = 0;
            unsigned num_threads                = 0;
            // A negative value disables the graceful wall-clock stop. When
            // enabled, no new true evaluation starts after this many seconds;
            // already-running OpenMP evaluations are allowed to drain.
            double wall_clock_limit_seconds     = -1.0;

            double priority_crossover_prob      = 0.3;
            double subacc_crossover_prob        = 0.3;
            double mapping_crossover_prob       = 0.3;

            double splitting_mutation_prob      = 0.15;
            double merging_mutation_prob        = 0.15;
            double priority_mutation_prob       = 0.15;
            double mapping_mutation_prob        = 0.15;
            double template_mutation_prob       = 0.10;
            double assignment_mutation_prob     = 0.15;
            double position_mutation_prob       = 0.15;

            bool xu_priority                    = true;
            bool use_tournament                 = false;
            bool random_when_illegal            = true;
            
            bool stop_on_convergence            = true;
            unsigned stability_window           = 5;
            double stability_threshold          = 0.04;

            unsigned max_per_workload_mappings  = 0;

            unsigned max_subaccelerators        = 0;
            double system_bandwidth             = -1.0;
            double nip_link_bandwidth           = -1.0;
            double nip_hop_energy               = -1.0;

            // Top-level evolutionary selection. The default keeps the original
            // MOHaM behaviour; c-nsga3 enables constrained NSGA-III selection.
            std::string algorithm               = "nsga2";
            std::string ctaea_second_parent_rule = "ca-internal";
            std::string ctaea_infeasible_tournament_rule = "archive-redraw";
            // auto: nsga2=3, nsga3/c-nsga3=4, c-nsga2/scba-nsga3=5. Use slo-risk
            // to run the required five-objective NSGA-II comparison baseline.
            std::string objective_mode          = "auto";
            unsigned reference_partitions       = 0;

            // A negative limit disables the corresponding physical constraint.
            double package_power_max            = -1.0;
            double chiplet_power_max            = -1.0;
            double package_area_max             = -1.0;
            bool enforce_bandwidth_constraints  = false;
            std::vector<double> workload_deadlines;

            // A static SLO is either supplied directly as a deadline, or
            // derived once from a fixed reference design: D_w = alpha*C_ref,w.
            // It must never depend on the candidate currently being evaluated.
            std::vector<double> slo_reference_latencies;
            double slo_multiplier               = -1.0;

            // SCBA-NSGA-III controls. A zero archive size selects one third
            // of the search population at construction time.
            unsigned boundary_archive_size      = 0;
            double boundary_parent_probability  = 0.25;
            bool adaptive_boundary_parenting    = false;
            // "profile" keeps a diverse Pareto set in the vector of
            // normalized violations. "scalar" is the equal-capacity control
            // that retains candidates by total normalized violation only.
            std::string boundary_archive_mode   = "profile";
            double reference_adaptation_rate    = 0.50;
            double bottleneck_repair_probability = 0.60;
            unsigned max_repair_attempts        = 2;
            // "guided" uses the bottleneck ledger. "random" keeps the same
            // trigger, action repertoire, acceptance rule, and true-evaluation
            // budget while removing bottleneck attribution from action choice.
            std::string repair_policy           = "guided";
            // Experimental attribution refinement. Disabled by default so
            // historical SCBA behavior remains reproducible.
            bool bandwidth_service_aware_repair = false;

            // Compact package RC model. Units are explicit: frequency in GHz,
            // energy in pJ, area in mm^2, resistance in K/W and capacitance in J/K.
            bool thermal_enabled                = false;
            double thermal_temperature_max      = -1.0;
            double thermal_validation_margin    = 0.0;
            double thermal_ambient_temperature  = 25.0;
            double thermal_frequency_ghz        = -1.0;
            double thermal_self_resistance      = -1.0;
            double thermal_neighbor_resistance  = -1.0;
            double thermal_capacitance_per_area = -1.0;
            double thermal_package_resistance   = -1.0;
            double thermal_package_capacitance  = -1.0;
            double thermal_time_step_cycles     = 1000.0;

            std::vector<std::pair<unsigned, unsigned>> memory_interfaces_amount;
            unsigned max_memory_interfaces_amount = 4;
            std::string memory_interfaces_position = "corner";

            bool explore_mapping                    = true;
            bool negotiate_arch                     = true;

            bool random_search                      = false;
            // Per-individual Pareto mapping files are large and redundant for
            // statistical sweeps. CSV metrics/schedules are always emitted.
            bool output_pareto_files                = true;

            bool single_obj                         = false;
            bool prod_obj                           = false;
            std::vector<double> weights_obj;
        } moham;

        std::string out_dir                     = ".";
        // Optional shared per-layer mapping cache. Keeping this independent
        // from out_dir lets all compared schedulers consume identical mappings.
        std::string mapping_cache_dir;
        // Explicitly trust serialized mapping objectives/architectures. The
        // default re-evaluates cache entries for backward compatibility.
        bool trust_mapping_cache               = false;
        unsigned long seed                      = 0;
    };

}


#endif // MOHAM_CONFIG_H_
