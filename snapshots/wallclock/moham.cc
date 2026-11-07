#include "moham.h"

#include <chrono>
#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>

#include "boost/filesystem.hpp"

#include "common.h"
#include "medea.h"
#include "timeloop.h"
#include "progressbar.h"
#include "archtemplate.h"
#include "graph.h"
#include "config.h"

namespace fs = boost::filesystem;

namespace moham 
{
    using MeshCoordinate = std::pair<int, int>;
    using MeshLink = std::pair<MeshCoordinate, MeshCoordinate>;

    static MeshLink CanonicalMeshLink(
        const MeshCoordinate& first, const MeshCoordinate& second)
    {
        return first < second ? MeshLink{first, second} : MeshLink{second, first};
    }

    static std::vector<MeshLink> BuildXYRoute(
        MeshCoordinate source, const MeshCoordinate& destination)
    {
        std::vector<MeshLink> route;
        while (source.first != destination.first)
        {
            MeshCoordinate next = source;
            next.first += destination.first > source.first ? 1 : -1;
            route.push_back(CanonicalMeshLink(source, next));
            source = next;
        }
        while (source.second != destination.second)
        {
            MeshCoordinate next = source;
            next.second += destination.second > source.second ? 1 : -1;
            route.push_back(CanonicalMeshLink(source, next));
            source = next;
        }
        return route;
    }

    Moham::Moham(Config config, Graph* graph, std::vector<ArchTemplate>* arch_templates, std::vector<std::unordered_map<ArchTemplate::ID,Medea::Population>> workload_mappings) :
        config_(config),
        graph_(graph),
        arch_templates_(arch_templates),
        workload_mappings_(workload_mappings),
        uni_distribution_(0, 1)
    {
        // Inizialization

        parent_population_.resize(config_.moham.population_size);
        population_.resize(config_.moham.population_size);
        immigrant_population_.resize(config_.moham.immigrant_population_size);
        merged_population_.resize(2 * config_.moham.population_size + config_.moham.immigrant_population_size);

        std::random_device rand_dev;
        auto seed = config.seed ? config.seed : rand_dev();
        for (int t = 0; t < omp_get_max_threads(); t++)
            rng_.emplace_back(seed + t);

        if (config.moham.num_threads > 0) {
            omp_set_num_threads(config_.moham.num_threads);
        }

        layer_network_ids_.reserve(graph_->NumLayers());
        for (Layer::ID layer_id = 0; layer_id < graph_->NumLayers(); ++layer_id)
        {
            const std::string& network = (*graph_)[layer_id].network;
            auto network_it = std::find(network_names_.begin(), network_names_.end(), network);
            if (network_it == network_names_.end())
            {
                network_names_.push_back(network);
                layer_network_ids_.push_back(network_names_.size() - 1);
            }
            else
            {
                layer_network_ids_.push_back(
                    static_cast<std::size_t>(std::distance(network_names_.begin(), network_it)));
            }
        }

        gen_stability_measures_.resize(config_.moham.stability_window);
        config_.moham.max_subaccelerators = config_.moham.max_subaccelerators > 0 ? config_.moham.max_subaccelerators : graph_->NumLayers();
        
        // Calculating mappings pareto fronts objective bounds for normalization in mapping crossover situations
        workload_mappings_objectives_bounds_.resize(workload_mappings_.size());
        workload_mappings_normalized_objectives_.resize(workload_mappings_.size());

        for (Workload::ID w = 0; w < workload_mappings_.size(); w++)
            for (const auto& template_mappings : workload_mappings_[w])
            {
                for (unsigned o = 0; o < workload_mappings_objectives_bounds_[w][template_mappings.first].size(); o++)
                {
                    auto& bounds = workload_mappings_objectives_bounds_[w][template_mappings.first][o];
                    bounds = std::make_pair(std::numeric_limits<double>::max(), 0.0);
                    for (const auto& ind : template_mappings.second)
                    {
                        bounds.first = std::min(bounds.first, ind.objectives[o]);
                        bounds.second = std::max(bounds.second, ind.objectives[o]);
                    }
                }

                workload_mappings_normalized_objectives_[w].emplace(template_mappings.first, template_mappings.second.size());
                for (Medea::Individual::ID i = 0; i < template_mappings.second.size(); i++)
                    for (unsigned o = 0; o < workload_mappings_normalized_objectives_[w][template_mappings.first][i].size(); o++)
                    {
                        const auto& bounds = workload_mappings_objectives_bounds_[w][template_mappings.first][o];
                        const double range = bounds.second - bounds.first;
                        workload_mappings_normalized_objectives_[w][template_mappings.first][i][o] =
                            range > 1e-12
                                ? (template_mappings.second[i].objectives[o] - bounds.first) / range
                                : 0.0;
                    }
            }
        
        // TODO Calculate scheduling space

        std::cout << "🦋 MOHAM Scheduling " << graph_->NumLayers() << " layers.";

        // TODO check that for each layer in the graph there is at least one template that is allowed to execute it.

        // Single Objective Stuff
        single_obj_ = config_.moham.single_obj;
        prod_obj_ = config_.moham.prod_obj;
        weights_obj_ = config_.moham.weights_obj;

        const bool scba_nsga3 = config_.moham.algorithm == "scba-nsga3";
        const bool constrained_nsga2 = config_.moham.algorithm == "c-nsga2";
        const bool ctaea = config_.moham.algorithm == "c-taea";
        const bool nsga3 = config_.moham.algorithm == "nsga3" || config_.moham.algorithm == "c-nsga3" || scba_nsga3;
        const bool constrained_nsga3 = config_.moham.algorithm == "c-nsga3" || scba_nsga3;
        const bool constrained_algorithm = constrained_nsga2 || constrained_nsga3 || ctaea;
        if (!nsga3 && !constrained_nsga2 && !ctaea &&
            config_.moham.algorithm != "nsga2")
            throw std::runtime_error("Unknown global-scheduler algorithm: " + config_.moham.algorithm);
        if (constrained_algorithm && !config_.moham.thermal_enabled)
            throw std::runtime_error("constrained scheduling requires architecture.thermal.enabled so peak temperature is evaluable");
        if (scba_nsga3)
        {
            if (config_.moham.boundary_parent_probability < 0.0 || config_.moham.boundary_parent_probability > 1.0 ||
                config_.moham.bottleneck_repair_probability < 0.0 || config_.moham.bottleneck_repair_probability > 1.0 ||
                config_.moham.reference_adaptation_rate < 0.0)
                throw std::runtime_error("SCBA probabilities must be in [0,1] and reference-adaptation-rate non-negative");
            if (config_.moham.boundary_archive_mode != "profile" &&
                config_.moham.boundary_archive_mode != "scalar")
                throw std::runtime_error(
                    "boundary-archive-mode must be profile or scalar");
            if (config_.moham.repair_policy != "guided" &&
                config_.moham.repair_policy != "random")
                throw std::runtime_error(
                    "repair-policy must be guided or random");
            const unsigned archive_size = config_.moham.boundary_archive_size
                ? config_.moham.boundary_archive_size : std::max(1u, config_.moham.population_size / 3);
            feasible_archive_.reserve(archive_size);
            boundary_archive_.reserve(archive_size);
        }
        if (ctaea)
        {
            ctaea_convergence_archive_.reserve(config_.moham.population_size);
            ctaea_diversity_archive_.reserve(config_.moham.population_size);
        }
        unsigned objective_count = (scba_nsga3 || ctaea)
            ? 5u : (nsga3 ? 4u : (constrained_nsga2 ? 5u : 3u));
        if (config_.moham.objective_mode == "legacy") objective_count = 3;
        else if (config_.moham.objective_mode == "thermal") objective_count = 4;
        else if (config_.moham.objective_mode == "slo-risk") objective_count = 5;
        else if (config_.moham.objective_mode != "auto")
            throw std::runtime_error("Unknown objective-mode: " + config_.moham.objective_mode);
        if (constrained_nsga2 && objective_count != 5)
            throw std::runtime_error("c-nsga2 is the five-objective SLO-Risk comparison baseline");
        if (scba_nsga3 && objective_count != 5)
            throw std::runtime_error("scba-nsga3 requires all five SLO-Risk objectives");
        if (ctaea && objective_count != 5)
            throw std::runtime_error("c-taea requires all five SLO-Risk objectives");
        if (objective_count == 5)
        {
            if (config_.moham.workload_deadlines.size() != network_names_.size())
                throw std::runtime_error("five-objective SLO-Risk scheduling requires one static deadline for every input DNN/network");
            if (std::any_of(config_.moham.workload_deadlines.begin(), config_.moham.workload_deadlines.end(),
                            [](double deadline) { return deadline <= 0.0; }))
                throw std::runtime_error("SLO deadlines must be positive");
        }
        ConfigureObjectiveCount(objective_count);
        ConfigureNSGAIII(
            nsga3, config_.moham.reference_partitions,
            constrained_algorithm && !ctaea);
    }


    bool Moham::WallClockExpired()
    {
        if (!wall_clock_limit_active_)
            return false;
        if (std::chrono::steady_clock::now() < wall_clock_deadline_)
            return false;
        wall_clock_limit_reached_.store(true, std::memory_order_relaxed);
        return true;
    }


    void Moham::Run()
    {  
        Timer search_timer; 
        search_timer.start();
        const auto controlled_start = std::chrono::steady_clock::now();
        wall_clock_limit_active_ = config_.moham.wall_clock_limit_seconds > 0.0;
        if (wall_clock_limit_active_)
            wall_clock_deadline_ = controlled_start + std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(
                        config_.moham.wall_clock_limit_seconds));
        wall_clock_limit_reached_.store(false, std::memory_order_relaxed);

        std::cout << "   MT Test - These numbers should be different: ";
        #pragma omp parallel for
        for (int i = 0; i < 6; i++) {
            #pragma omp critical
            std::cout << (std::size_t)problem::GetShape() % sizeof(timeloop::Shape) << " ";
        }
        std::cout << std::endl;

        std::string moham_out_path = config_.out_dir + "/moham/";
        fs::create_directories(moham_out_path);
        if (config_.moham.output_pareto_files)
            fs::create_directories(moham_out_path + "pareto");

        const std::string pop_filename = "moham.populations.csv";
        std::ofstream population_file(moham_out_path + pop_filename);

        const std::string eval_filename = "moham.evaluations.csv";
        std::ofstream evaluations_file(moham_out_path + eval_filename);
        evaluations_file << "generation,rank,selection_metric,energy_pj,makespan_cycles,area_um2,peak_temperature_c,slo_risk,peak_package_power_w,peak_chiplet_power_w,peak_system_bandwidth,peak_nip_bandwidth,network_completion_cycles,compute_only_makespan_cycles,bandwidth_makespan_overhead_cycles,bandwidth_makespan_overhead_ratio,bandwidth_service_stretch_cycles,bandwidth_queue_delay_cycles,system_bandwidth_queue_delay_cycles,nip_bandwidth_queue_delay_cycles,bandwidth_stretched_layers,bandwidth_delayed_layers,constraint_profile,constraint_violation,feasible" << std::endl;

        // === INITIAL POPULATION ===

        double rpt = RandomPopulation(parent_population_);
        InjectHeuristicallyGoodIndividuals(parent_population_);
        AssignRankAndCrowdingDistance(parent_population_);
        AppendEvaluationInfoToFile(evaluations_file, parent_population_, 0);
        if (config_.moham.algorithm == "scba-nsga3")
        {
            UpdateSCBAArchives(parent_population_);
            UpdateReferenceDirectionBias();
        }
        else if (config_.moham.algorithm == "c-taea")
        {
            UpdateCTAEAArchives(parent_population_);
            parent_population_ = ctaea_convergence_archive_;
            UpdateCTAEAMatingProbabilities();
        }
        std::cout << "   Initial | " << std::fixed << std::setprecision(2) << rpt << " s" << std::endl;

        // === MAIN LOOP ===

        bool convergence_flag = false;
        unsigned completed_generations = 0;

        for (unsigned g = 0;
             !convergence_flag && !WallClockExpired() &&
             (g < config_.moham.num_generations && config_.moham.num_generations);
             g++)
        {
            if (config_.moham.algorithm == "scba-nsga3")
                UpdateSCBABoundaryPolicy();

            const unsigned configured_immigrants = config_.moham.immigrant_population_size;
            repair_evaluation_budget_.store(
                config_.moham.algorithm == "scba-nsga3" ? configured_immigrants : 0,
                std::memory_order_relaxed);

            std::stringstream ss;
            ss << "   G " << std::right << std::setfill('0') << std::setw(3) << g << "   ";

            unsigned success_cross_count = 0;
            
            ProgressBar bar;
            bar.start(population_.size(), ss.str());

            #pragma omp parallel for schedule(static)
            for (unsigned i = 0; i < population_.size(); i += 2)
            {
                if (config_.moham.algorithm == "scba-nsga3")
                {
                    Crossover(SelectSCBAParent(), SelectSCBAParent(), population_[i], population_[i+1]);
                }
                else if (config_.moham.algorithm == "c-taea")
                {
                    Crossover(
                        SelectCTAEAParent(true), SelectCTAEAParent(false),
                        population_[i], population_[i+1]);
                }
                else
                {
                    std::size_t parent_a = config_.moham.use_tournament ? Tournament(parent_population_) : i;
                    std::size_t parent_b = config_.moham.use_tournament ? Tournament(parent_population_) : i + 1;
                    Crossover(parent_population_[parent_a], parent_population_[parent_b], population_[i], population_[i+1]);
                }

                // Crossover copies parent metadata. Clear validity so only
                // offspring evaluated before the common wall-clock deadline
                // can enter environmental selection.
                population_[i].valid = false;
                population_[i + 1].valid = false;

                for (unsigned j = 0; j < 2; j++) 
                {
                    if (WallClockExpired())
                    {
                        ++bar;
                        continue;
                    }
                    Mutation(population_[i+j]);

                    if ( CheckIndividualValidity(population_[i+j]) ) {
                        #pragma omp atomic update
                        success_cross_count++;
                        Evaluate(population_[i+j]);
                    } 
                    else if (config_.moham.random_when_illegal) {
                        do {
                            population_[i+j] = RandomIndividual();
                        } while (!CheckIndividualValidity(population_[i+j], false));
                        Evaluate(population_[i+j]);
                    } 

                    ++bar;
                }
            }
            bar.stop();

            // Repair in population order after parallel offspring evaluation.
            // The previous atomic "first thread to finish" policy made a fixed
            // seed depend on OS/OpenMP timing and changed which individuals
            // consumed the shared repair budget.
            if (config_.moham.algorithm == "scba-nsga3")
                for (auto& individual : population_)
                {
                    if (WallClockExpired())
                        break;
                    if (repair_evaluation_budget_.load(std::memory_order_relaxed) == 0)
                        break;
                    if (individual.valid)
                        ApplyBottleneckGuidedRepair(individual, rng_[0]);
                }

            const unsigned unused_repair_budget =
                repair_evaluation_budget_.load(std::memory_order_relaxed);
            immigrant_population_.resize(
                config_.moham.algorithm == "scba-nsga3"
                    ? unused_repair_budget
                    : configured_immigrants);
            RandomPopulation(immigrant_population_);
            AppendEvaluationInfoToFile(evaluations_file, population_, g + 1);
            AppendEvaluationInfoToFile(evaluations_file, immigrant_population_, g + 1);

            if (config_.moham.algorithm == "c-taea")
            {
                Population ctaea_candidates = immigrant_population_;
                std::copy_if(
                    population_.begin(), population_.end(),
                    std::back_inserter(ctaea_candidates),
                    [](const auto& individual) { return individual.valid; });
                UpdateCTAEAArchives(ctaea_candidates);
                parent_population_ = ctaea_convergence_archive_;
                UpdateCTAEAMatingProbabilities();
            }
            else
            {
                Merging(merged_population_, parent_population_, population_, immigrant_population_);
                AssignRankAndCrowdingDistance(merged_population_);
                if (config_.moham.algorithm == "scba-nsga3")
                {
                    UpdateSCBAArchives(merged_population_);
                    UpdateReferenceDirectionBias();
                }
                Survival(parent_population_, merged_population_, !config_.moham.use_tournament, rng_[0]);
            }

            // Print Gen Info   
            std::cout << ss.str();
            ss.str(""); ss << "| " << std::fixed << std::setprecision(2) << bar.time_it_took() << " s";
            std::cout << std::setw(13) << ss.str();
            ss.str(""); ss << "| " << success_cross_count << "/" << std::left << population_.size();
            std::cout << std::setw(13) << ss.str() << "| R  " << RankString(parent_population_);

            // Stopping Criterion Check - Implementation of https://hal.inria.fr/hal-01909120/document Eq. 2
            if (config_.moham.stop_on_convergence) 
            {
                UpdateStabilityMeasure(parent_population_, g, config_.medea.stability_window);
                if (g >= config_.moham.stability_window)
                { 
                    stability_ = CalculateGenerationStability(config_.moham.stability_window);
                    convergence_flag = (stability_ < config_.moham.stability_threshold);
                    std::cout << " | S " << stability_;
                } else {
                    std::cout << " | S n/a";
                }
            }

            std::cout << std::endl;
            AppendGenerationInfoToFile(population_file, parent_population_, g);
            ++completed_generations;
        }

        const double optimization_elapsed_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - controlled_start).count();

        // === TERMINATION ===

        if (convergence_flag) 
            std::cout << "   Stability detected." << std::endl;

        population_file.close();
        evaluations_file.close();

        const std::string ledger_filename = "moham.evaluation_ledger.csv";
        std::ofstream ledger_file(moham_out_path + ledger_filename);
        OutputEvaluationLedger(ledger_file);
        ledger_file.close();

        const std::string final_population_filename = "moham.final_population_metrics.csv";
        std::ofstream final_population_file(moham_out_path + final_population_filename);
        OutputFinalPopulationMetrics(final_population_file, parent_population_);
        final_population_file.close();

        const std::string pareto_filename = "moham.pareto_schedulings.csv";
        std::ofstream pareto_file(moham_out_path + pareto_filename);
        OutputParetoScheduling(pareto_file, parent_population_);
        pareto_file.close();

        const std::string nip_bottle_filename = "moham.pareto_resource_bottlenecks.csv";
        std::ofstream pareto_nip_file(moham_out_path + nip_bottle_filename);
        OutputParetoNipBottlenecks(pareto_nip_file, parent_population_);
        pareto_nip_file.close();

        const std::string metrics_filename = "moham.pareto_metrics.csv";
        std::ofstream metrics_file(moham_out_path + metrics_filename);
        OutputParetoMetrics(metrics_file, parent_population_);
        metrics_file.close();

        double search_time = search_timer.stop();

        const std::string stats_filename = "moham.stats.txt";
        std::ofstream stats_file(moham_out_path + stats_filename);
        stats_file << "Search time: " << search_time << " seconds" << std::endl;
        stats_file << "Optimization wall time: " << optimization_elapsed_seconds
                   << " seconds" << std::endl;
        stats_file << "Wall clock limit: "
                   << config_.moham.wall_clock_limit_seconds << " seconds"
                   << std::endl;
        stats_file << "Wall clock limit reached: "
                   << (wall_clock_limit_reached_.load(std::memory_order_relaxed)
                           ? "true" : "false")
                   << std::endl;
        stats_file << "Wall clock drain overshoot: "
                   << (wall_clock_limit_active_
                           ? std::max(
                                 0.0,
                                 optimization_elapsed_seconds -
                                     config_.moham.wall_clock_limit_seconds)
                           : 0.0)
                   << " seconds" << std::endl;
        stats_file << "Completed generations: " << completed_generations
                   << std::endl;
        stats_file << "Seed: " << config_.seed << std::endl;
        stats_file << "Algorithm: " << config_.moham.algorithm << std::endl;
        stats_file << "Objective mode: " << config_.moham.objective_mode << std::endl;
        stats_file << "Boundary archive mode: "
                   << config_.moham.boundary_archive_mode << std::endl;
        stats_file << "Repair policy: "
                   << config_.moham.repair_policy << std::endl;
        stats_file << "Evaluations: " << evaluation_count_.load() << std::endl;
        stats_file << "Repair evaluations: " << repair_evaluation_count_.load() << std::endl;
        stats_file << "Accepted repair evaluations: "
                   << accepted_repair_evaluation_count_.load() << std::endl;
        stats_file << "Bandwidth service-aware repair: "
                   << (config_.moham.bandwidth_service_aware_repair ? "true" : "false")
                   << std::endl;
        const auto scba_parent_selections = scba_parent_selection_count_.load();
        const auto scba_boundary_selections =
            scba_boundary_parent_selection_count_.load();
        stats_file << "Adaptive boundary parenting: "
                   << (config_.moham.adaptive_boundary_parenting ? "true" : "false")
                   << std::endl;
        stats_file << "Boundary parent probability max: "
                   << config_.moham.boundary_parent_probability << std::endl;
        stats_file << "SCBA parent selections: " << scba_parent_selections
                   << std::endl;
        stats_file << "SCBA boundary parent selections: "
                   << scba_boundary_selections << std::endl;
        stats_file << "SCBA forced boundary parent selections: "
                   << scba_forced_boundary_parent_selection_count_.load()
                   << std::endl;
        stats_file << "SCBA boundary parent selection rate: "
                   << (scba_parent_selections
                           ? static_cast<double>(scba_boundary_selections) /
                                 scba_parent_selections
                           : 0.0)
                   << std::endl;
        stats_file << "SCBA mean effective boundary probability: "
                   << (scba_parent_selections
                           ? static_cast<double>(
                                 scba_effective_probability_sum_ppm_.load()) /
                                 (1.0e6 * scba_parent_selections)
                           : 0.0)
                   << std::endl;
        stats_file << "First feasible evaluation: " << first_feasible_evaluation_.load() << std::endl;
        const auto feasible_evaluations = std::count_if(
            evaluation_records_.begin(), evaluation_records_.end(),
            [](const EvaluationRecord& record) {
                return record.constraint_violation <= 1e-12;
            });
        std::unordered_set<std::string> unique_feasible_designs;
        for (const auto& record : evaluation_records_)
            if (record.constraint_violation <= 1e-12)
                unique_feasible_designs.insert(record.design_fingerprint);
        stats_file << "Feasible evaluations: " << feasible_evaluations << std::endl;
        stats_file << "Unique feasible designs: " << unique_feasible_designs.size() << std::endl;
        stats_file << "Input DNNs: " << network_names_.size() << std::endl;
        stats_file << "DNN names:";
        for (const auto& network_name : network_names_)
            stats_file << " " << network_name;
        stats_file << std::endl;
        stats_file << "SLO deadlines:";
        for (double deadline : config_.moham.workload_deadlines)
            stats_file << " " << deadline;
        stats_file << std::endl;
        stats_file << "Mapping cache: "
                   << (config_.mapping_cache_dir.empty() ? "run-local" : config_.mapping_cache_dir)
                   << std::endl;
        stats_file.close();

        if (config_.moham.output_pareto_files)
            OutputParetoFrontFiles(moham_out_path + "pareto");

        std::cout << "   Search time: " << search_time << " seconds" << std::endl;
    }


    void Moham::RunRandom() 
    {
        Timer search_timer; 
        search_timer.start();

        std::cout << "   RANDOM SEARCH MODE" << std::endl;
        std::cout << "   MT Test - These numbers should be different: ";
        #pragma omp parallel for
        for (int i = 0; i < 6; i++) {
            #pragma omp critical
            std::cout << (std::size_t)problem::GetShape() % sizeof(timeloop::Shape) << " ";
        }
        std::cout << std::endl;

        std::string moham_out_path = config_.out_dir + "/moth/";
        fs::create_directories(moham_out_path);

        const std::string eval_filename = "moham.evaluations.csv";
        std::ofstream evaluations_file(moham_out_path + eval_filename);
        evaluations_file << "generation,rank,selection_metric,energy_pj,makespan_cycles,area_um2,peak_temperature_c,slo_risk,peak_package_power_w,peak_chiplet_power_w,peak_system_bandwidth,peak_nip_bandwidth,network_completion_cycles,compute_only_makespan_cycles,bandwidth_makespan_overhead_cycles,bandwidth_makespan_overhead_ratio,bandwidth_service_stretch_cycles,bandwidth_queue_delay_cycles,system_bandwidth_queue_delay_cycles,nip_bandwidth_queue_delay_cycles,bandwidth_stretched_layers,bandwidth_delayed_layers,constraint_profile,constraint_violation,feasible" << std::endl;

        for (unsigned g = 0; g < config_.moham.num_generations; g++) 
        { 
            double rpt = RandomPopulation(parent_population_);
            AppendEvaluationInfoToFile(evaluations_file, parent_population_, g);
            rpt += RandomPopulation(immigrant_population_);
            AppendEvaluationInfoToFile(evaluations_file, immigrant_population_, g);

            std::cout << "   Random " << std::right << std::setfill('0') << std::setw(3) << g << " | " << std::fixed << std::setprecision(2) << rpt << " s" << std::endl;
        }
        
        evaluations_file.close();

        double search_time = search_timer.stop();

        const std::string stats_filename = "moham.stats.txt";
        std::ofstream stats_file(moham_out_path + stats_filename);
        stats_file << "Search time: " << search_time << " seconds" << std::endl;
        stats_file.close();

        std::cout << "   Search time: " << search_time << " seconds" << std::endl;
    }


    double Moham::RandomPopulation(Population& pop) 
    {
        ProgressBar bar;
        bar.start(pop.size(), "   ");

        #pragma omp parallel for schedule(static)
        for (std::size_t p = 0; p < pop.size(); p++) 
        {
            pop[p].valid = false;
            if (WallClockExpired())
            {
                ++bar;
                continue;
            }
            do {
                pop[p] = RandomIndividual();
            } while (!CheckIndividualValidity(pop[p], false));

            Evaluate(pop[p]);
            ++bar;
        }

        bar.stop();
        pop.erase(
            std::remove_if(
                pop.begin(), pop.end(),
                [](const auto& individual) { return !individual.valid; }),
            pop.end());
        return bar.time_it_took();
    }


    Moham::Individual Moham::RandomIndividual() 
    {
        auto& rng =  rng_[omp_get_thread_num()];
        std::size_t num_layers = graph_->NumLayers();

        Individual ind;
        ind.layer_evals.resize(num_layers, {0,0,0,0,0,0,true});

        // Choosing how many subaccs
        unsigned num_accs = rng() % config_.moham.max_subaccelerators;
        if (num_accs == 0) num_accs = 1;
        ind.accelerators.resize(num_accs);

        // Choosing subaccs templates
        ArchInstance::ID instance_id = 0;
        for (auto& acc : ind.accelerators) 
        {
            acc.id = instance_id++;
            acc.arch_template_id = rng() % arch_templates_->size();
            acc.num_assigned_layers = 0;
        }

        ind.genome.resize(num_layers);

        for (std::size_t l = 0; l < num_layers; l++) 
        {
            // Try 5 times to find acc instance that allows this type of layer
            // validity check will do the rest
            for (unsigned tries = 0; tries < 5; tries++) 
            {
                ArchInstance::ID acc_instance_id = rng() % num_accs;
                ArchTemplate::ID template_id = ind.accelerators[acc_instance_id].arch_template_id;

                std::size_t possible_mappings = workload_mappings_[(*graph_)[l].workload_id].at(template_id).size();
                Medea::Individual::ID mapping_id = possible_mappings > 0 ? rng() % possible_mappings : 0;
                
                ind.genome[l] = (Gene){
                    .priority =         uni_distribution_(rng),
                    .mapping_id =       mapping_id,
                    .acc_instance_id =  acc_instance_id
                };

                if (possible_mappings > 0) {
                    ind.accelerators[acc_instance_id].num_assigned_layers++;
                    break;
                }
            }
        }

        //AssignBreadthFirstPriorities(ind);

        return ind;   
    }


    void Moham::InjectHeuristicallyGoodIndividuals(Population& population) 
    {
        if (population.empty()) return;
        std::size_t num_layers = graph_->NumLayers();

        // Lowest latency
        {
            const std::size_t individual_id = 0;
            Individual& chosen_individual = population[individual_id];
            chosen_individual.accelerators.clear();

            for (Layer::ID l = 0; l < num_layers; l++)
            {
                ArchTemplate::ID min_template = 0;
                Medea::Individual::ID min_mapping = 0;
                double min_latency = std::numeric_limits<double>::max();

                for (auto& template_mappings : workload_mappings_[(*graph_)[l].workload_id])
                    for (Medea::Individual::ID m = 0; m < template_mappings.second.size(); m++)
                        if (template_mappings.second[m].objectives[1] < min_latency)
                        {
                            min_template = template_mappings.first;
                            min_mapping = m;
                            min_latency = template_mappings.second[m].objectives[1];
                        }

                chosen_individual.genome[l].mapping_id = min_mapping;
                auto accelerator = std::find_if(chosen_individual.accelerators.begin(), chosen_individual.accelerators.end(), 
                                                [&](const ArchInstance& acc){ return acc.arch_template_id == min_template; });
                if (accelerator == chosen_individual.accelerators.end())
                {
                    ArchInstance::ID aid = chosen_individual.accelerators.size();
                    chosen_individual.accelerators.emplace_back();
                    chosen_individual.accelerators.back().id = aid;
                    chosen_individual.accelerators.back().arch_template_id = min_template;  
                    chosen_individual.accelerators.back().num_assigned_layers = 1;     
                    chosen_individual.genome[l].acc_instance_id = aid;
                } else {
                    chosen_individual.genome[l].acc_instance_id = accelerator->id;
                    accelerator->num_assigned_layers++;
                }
            }

            // Breadth-First topological ordering
            AssignBreadthFirstPriorities(chosen_individual);

            assert(CheckIndividualValidity(chosen_individual));
            Evaluate(chosen_individual);
        }

        // Lowest energy
        {
            const std::size_t individual_id = population.size() > 1 ? 1 : 0;
            Individual& chosen_individual = population[individual_id];
            chosen_individual.accelerators.clear();

            for (Layer::ID l = 0; l < num_layers; l++)
            {
                ArchTemplate::ID min_template = 0;
                Medea::Individual::ID min_mapping = 0;
                double min_energy = std::numeric_limits<double>::max();

                for (auto& template_mappings : workload_mappings_[(*graph_)[l].workload_id])
                    for (Medea::Individual::ID m = 0; m < template_mappings.second.size(); m++)
                        if (template_mappings.second[m].objectives[0] < min_energy)
                        {
                            min_template = template_mappings.first;
                            min_mapping = m;
                            min_energy = template_mappings.second[m].objectives[0];
                        }

                chosen_individual.genome[l].mapping_id = min_mapping;
                auto accelerator = std::find_if(chosen_individual.accelerators.begin(), chosen_individual.accelerators.end(), 
                                                [&](const ArchInstance& acc){ return acc.arch_template_id == min_template; });
                if (accelerator == chosen_individual.accelerators.end())
                {
                    ArchInstance::ID aid = chosen_individual.accelerators.size();
                    chosen_individual.accelerators.emplace_back();
                    chosen_individual.accelerators.back().id = aid;
                    chosen_individual.accelerators.back().arch_template_id = min_template; 
                    chosen_individual.accelerators.back().num_assigned_layers = 1;     
                    chosen_individual.genome[l].acc_instance_id = aid;
                } else {
                    chosen_individual.genome[l].acc_instance_id = accelerator->id;
                    accelerator->num_assigned_layers++;
                }
            }

            AssignBreadthFirstPriorities(chosen_individual);

            assert(CheckIndividualValidity(chosen_individual));
            Evaluate(chosen_individual);
        }

    }


    void Moham::Evaluate(Individual& ind, bool repair)
    {
        const unsigned long long evaluation_id =
            evaluation_count_.fetch_add(1, std::memory_order_relaxed) + 1;
        NegotiateArchitectures(ind);

        for (Layer::ID l = 0; l < graph_->NumLayers(); l++)
        {
            const Layer& layer = (*graph_)[l];
            const Gene& gene = ind.genome[l];
            ArchInstance& accelerator = ind.accelerators[gene.acc_instance_id];

            // Evaluate if new layer mapping or instance architecture updated
            //if (ind.layer_evals[l].need_evaluation || accelerator.need_evaluation) 
            { 
                auto engine = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].engine;
                auto& mapping = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].genome;
                auto& workload = graph_->GetWorkloads()[layer.workload_id]; 
                auto* sparse_opt = &(*arch_templates_)[accelerator.arch_template_id].sparse_optimizations;
                
                problem::shape_ = workload.shape; // Update thread private global variable before evaluating
                
                engine.Spec(accelerator.arch_specs);
                auto status = engine.Evaluate(mapping, workload.workload, sparse_opt);

                if (!engine.IsEvaluated())
                {
                    std::cout << "Error in workload " << (*graph_)[l].workload_id << " with mapping " << ind.genome[l].mapping_id << std::endl;

                    for (auto &s : status)
                        std::cout << s.success << " " << s.fail_reason << std::endl;
                    YAML::Emitter yout;
                    for (auto& a : ind.accelerators)
                        yout << a.arch;
                    std::cout << yout.c_str() << std::endl;
                    exit(1);
                } 
                
                ind.layer_evals[l].energy = engine.Energy();
                ind.layer_evals[l].cycles = engine.Cycles();

                // Update Bandwidth Requirement
                const timeloop::Topology& topology = engine.GetTopology();
                auto arithmetic_minarch = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].arch.GetLevel(0);
                ind.layer_evals[l].required_bandwidth = (double)topology.LastLevelAccesses() / ((double)topology.ActualComputes() / (arithmetic_minarch.mesh_x * arithmetic_minarch.mesh_y));
            
                // Accelerator area
                accelerator.area = engine.Area();
                ind.layer_evals[l].area = engine.Area();

                // Evaluated
                ind.layer_evals[l].need_evaluation = false;
            }
        }

        for (auto& acc : ind.accelerators)
            acc.need_evaluation = false;
        
        // Evaluating Objectives

        std::fill(ind.objectives.begin(), ind.objectives.end(), 0.0);
        const bool legacy_moham_selection = config_.moham.algorithm == "nsga2" &&
            (config_.moham.objective_mode == "auto" || config_.moham.objective_mode == "legacy");
        const bool evaluate_physical_metrics =
            config_.moham.thermal_enabled ||
            config_.moham.package_power_max > 0.0 ||
            config_.moham.chiplet_power_max > 0.0 ||
            config_.moham.package_area_max > 0.0 ||
            config_.moham.enforce_bandwidth_constraints ||
            !config_.moham.workload_deadlines.empty();

        // Package placement is needed by the communication, power and thermal models.
        if (evaluate_physical_metrics || config_.moham.nip_hop_energy > 0.0 || config_.moham.nip_link_bandwidth > 0.0)
            CalculatePackageMesh(ind);

        // Energy
        for (const auto& eval : ind.layer_evals)
            ind.objectives[0] += eval.energy;

        // Total NiP Hop Energy
        ind.objectives[0] += EvaluateNipEnergy(ind);

        // Makespan
        if (!config_.moham.xu_priority || ind.toposort.empty())
            ind.toposort = GenerateTopologicalSorting(ind);

        ind.objectives[1] = EvaluateMakespan(ind, ind.toposort);

        // Area
        if (config_.moham.negotiate_arch) {
            for (const auto& acc : ind.accelerators) 
                ind.objectives[2] += acc.area;
        } else {
            ind.objectives[2] = config_.moham.max_subaccelerators * ind.accelerators[0].area;
        }

        if (evaluate_physical_metrics)
        {
            EvaluatePhysicalMetrics(ind);
            ind.objectives[3] = ind.peak_temperature;
            EvaluateConstraints(ind);
            ind.objectives[4] = 0.0;
            for (std::size_t network_id = 0;
                 network_id < ind.network_completion_times.size() && network_id < config_.moham.workload_deadlines.size();
                 ++network_id)
            {
                const double deadline = config_.moham.workload_deadlines[network_id];
                if (deadline > 0.0)
                    ind.objectives[4] = std::max(ind.objectives[4], ind.network_completion_times[network_id] / deadline);
            }
            BuildBottleneckLedger(ind);
        }
        else
        {
            // Preserve original three-objective MOHaM evaluation semantics
            // when no diagnostic physical/SLO model is configured.
            ind.constraint_profile.fill(0.0);
            ind.constraint_violation = 0.0;
            ind.bottleneck_ledger.clear();
        }

        // Selection remains exactly three-objective and unconstrained for the
        // legacy baseline; objectives 4/5 and violations above are diagnostics.
        (void) legacy_moham_selection;

        if (ind.constraint_violation <= 1e-12)
        {
            unsigned long long current =
                first_feasible_evaluation_.load(std::memory_order_relaxed);
            while ((current == 0 || evaluation_id < current) &&
                   !first_feasible_evaluation_.compare_exchange_weak(
                       current, evaluation_id, std::memory_order_relaxed))
            {
            }
        }

        ind.evaluation_id = evaluation_id;
        ind.design_fingerprint = DesignFingerprint(ind);
        RecordEvaluation(ind, repair);
    }


    std::string Moham::DesignFingerprint(const Individual& ind) const
    {
        struct CanonicalAccelerator {
            ArchInstance::ID original_id;
            std::string description;
        };

        std::vector<CanonicalAccelerator> canonical_accelerators;
        canonical_accelerators.reserve(ind.accelerators.size());
        for (const auto& accelerator : ind.accelerators)
        {
            std::ostringstream description;
            description << accelerator.arch_template_id << ":"
                        << accelerator.mesh_coordinates.first << ":"
                        << accelerator.mesh_coordinates.second << ":"
                        << accelerator.memory_interface_id;
            for (std::size_t level = 0; level < accelerator.arch.NumLevels(); ++level)
            {
                const auto specs = accelerator.arch.GetLevel(level);
                description << ":" << specs.mesh_x << ":" << specs.mesh_y
                            << ":" << specs.size;
            }
            canonical_accelerators.push_back(
                {accelerator.id, description.str()});
        }
        std::stable_sort(
            canonical_accelerators.begin(), canonical_accelerators.end(),
            [](const auto& left, const auto& right) {
                if (left.description != right.description)
                    return left.description < right.description;
                return left.original_id < right.original_id;
            });

        std::unordered_map<ArchInstance::ID, std::size_t> canonical_id;
        std::ostringstream canonical;
        canonical << "mesh:" << ind.mesh_dimensions.first << ":"
                  << ind.mesh_dimensions.second << "|accelerators:"
                  << canonical_accelerators.size();
        for (std::size_t id = 0; id < canonical_accelerators.size(); ++id)
        {
            canonical_id[canonical_accelerators[id].original_id] = id;
            canonical << "|" << canonical_accelerators[id].description;
        }
        canonical << "|genome:" << ind.genome.size();
        for (const auto& gene : ind.genome)
            canonical << "|" << gene.mapping_id << ":"
                      << canonical_id.at(gene.acc_instance_id);
        canonical << "|schedule:" << ind.toposort.size();
        for (const auto layer_id : ind.toposort)
            canonical << "|" << layer_id;

        const std::string value = canonical.str();
        auto fnv1a = [&value](std::uint64_t state) {
            constexpr std::uint64_t prime = 1099511628211ULL;
            for (const unsigned char byte : value)
            {
                state ^= static_cast<std::uint64_t>(byte);
                state *= prime;
            }
            return state;
        };
        const std::uint64_t first = fnv1a(14695981039346656037ULL);
        const std::uint64_t second = fnv1a(7809847782465536322ULL);
        std::ostringstream fingerprint;
        fingerprint << std::hex << std::setfill('0')
                    << std::setw(16) << first << std::setw(16) << second;
        return fingerprint.str();
    }


    void Moham::RecordEvaluation(const Individual& ind, bool repair)
    {
        EvaluationRecord record;
        record.evaluation_id = ind.evaluation_id;
        record.design_fingerprint = ind.design_fingerprint;
        record.repair = repair;
        record.objectives = ind.objectives;
        record.peak_package_power = ind.peak_package_power;
        record.peak_chiplet_power = ind.peak_chiplet_power;
        record.peak_system_bandwidth = ind.peak_system_bandwidth;
        record.peak_nip_bandwidth = ind.peak_nip_bandwidth;
        record.compute_only_makespan = ind.compute_only_makespan;
        record.bandwidth_makespan_overhead = ind.bandwidth_makespan_overhead;
        record.bandwidth_service_stretch_cycles = ind.bandwidth_service_stretch_cycles;
        record.bandwidth_queue_delay_cycles = ind.bandwidth_queue_delay_cycles;
        record.system_bandwidth_queue_delay_cycles = ind.system_bandwidth_queue_delay_cycles;
        record.nip_bandwidth_queue_delay_cycles = ind.nip_bandwidth_queue_delay_cycles;
        record.bandwidth_stretched_layers = ind.bandwidth_stretched_layers;
        record.bandwidth_delayed_layers = ind.bandwidth_delayed_layers;
        record.network_completion_times = ind.network_completion_times;
        record.constraint_profile = ind.constraint_profile;
        record.constraint_violation = ind.constraint_violation;
        std::lock_guard<std::mutex> lock(evaluation_records_mutex_);
        evaluation_records_.push_back(std::move(record));
    }


    double Moham::EvaluatePeakBandwidth(const Individual& ind, bool per_memory_interface) const
    {
        const auto effective_bandwidth = [](const LayerEval& evaluation) {
            const double scheduled_cycles = std::max(
                evaluation.end_time - evaluation.start_time, 1e-12);
            const double transferred_data =
                evaluation.cycles * evaluation.required_bandwidth;
            return transferred_data / scheduled_cycles;
        };

        std::vector<double> boundaries;
        boundaries.reserve(ind.layer_evals.size() * 2);
        for (const auto& evaluation : ind.layer_evals)
        {
            boundaries.push_back(evaluation.start_time);
            boundaries.push_back(evaluation.end_time);
        }
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

        double peak_bandwidth = 0.0;
        for (std::size_t interval = 0; interval + 1 < boundaries.size(); ++interval)
        {
            const double start = boundaries[interval];
            const double end = boundaries[interval + 1];
            if (end <= start) continue;

            if (!per_memory_interface)
            {
                double required_bandwidth = 0.0;
                for (const auto& evaluation : ind.layer_evals)
                    if (evaluation.start_time <= start && end <= evaluation.end_time)
                        required_bandwidth += effective_bandwidth(evaluation);
                peak_bandwidth = std::max(peak_bandwidth, required_bandwidth);
            }
            else
            {
                std::map<MeshLink, double> required_bandwidth;
                for (Layer::ID layer_id = 0; layer_id < graph_->NumLayers(); ++layer_id)
                {
                    const auto& evaluation = ind.layer_evals[layer_id];
                    if (evaluation.start_time <= start && end <= evaluation.end_time)
                    {
                        const auto& accelerator =
                            ind.accelerators[ind.genome[layer_id].acc_instance_id];
                        if (accelerator.memory_interface_id >= ind.memory_interfaces.size())
                            continue;
                        const auto route = BuildXYRoute(
                            ind.memory_interfaces[accelerator.memory_interface_id].mesh_coordinates,
                            accelerator.mesh_coordinates);
                        for (const auto& link : route)
                            required_bandwidth[link] += effective_bandwidth(evaluation);
                    }
                }
                for (const auto& link_bandwidth : required_bandwidth)
                    peak_bandwidth = std::max(peak_bandwidth, link_bandwidth.second);
            }
        }
        return peak_bandwidth;
    }


    void Moham::EvaluatePhysicalMetrics(Individual& ind)
    {
        ind.peak_package_power = 0.0;
        ind.peak_chiplet_power.assign(ind.accelerators.size(), 0.0);
        ind.network_completion_times.assign(network_names_.size(), 0.0);
        for (Layer::ID layer_id = 0; layer_id < graph_->NumLayers(); ++layer_id)
        {
            const auto network_id = layer_network_ids_[layer_id];
            ind.network_completion_times[network_id] = std::max(
                ind.network_completion_times[network_id], ind.layer_evals[layer_id].end_time);
        }

        if (config_.moham.thermal_frequency_ghz > 0.0)
        {
            std::vector<double> boundaries;
            boundaries.reserve(ind.layer_evals.size() * 2);
            for (const auto& evaluation : ind.layer_evals)
            {
                boundaries.push_back(evaluation.start_time);
                boundaries.push_back(evaluation.end_time);
            }
            std::sort(boundaries.begin(), boundaries.end());
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

            for (std::size_t interval = 0; interval + 1 < boundaries.size(); ++interval)
            {
                const double start = boundaries[interval];
                const double end = boundaries[interval + 1];
                if (end <= start) continue;

                std::vector<double> chiplet_power(ind.accelerators.size(), 0.0);
                for (Layer::ID layer_id = 0; layer_id < graph_->NumLayers(); ++layer_id)
                {
                    const auto& evaluation = ind.layer_evals[layer_id];
                    if (evaluation.start_time <= start && end <= evaluation.end_time)
                    {
                        const double execution_cycles = std::max(evaluation.end_time - evaluation.start_time, 1e-12);
                        // pJ/cycle * GHz = mW; convert mW to W.
                        const double interval_energy =
                            evaluation.energy + EvaluateLayerCommunicationEnergy(ind, layer_id);
                        const double average_power_w = interval_energy / execution_cycles *
                            config_.moham.thermal_frequency_ghz * 1e-3;
                        chiplet_power[ind.genome[layer_id].acc_instance_id] += average_power_w;
                    }
                }

                double package_power = 0.0;
                for (ArchInstance::ID acc_id = 0; acc_id < chiplet_power.size(); ++acc_id)
                {
                    package_power += chiplet_power[acc_id];
                    ind.peak_chiplet_power[acc_id] = std::max(ind.peak_chiplet_power[acc_id], chiplet_power[acc_id]);
                }
                ind.peak_package_power = std::max(ind.peak_package_power, package_power);
            }
        }

        ind.peak_system_bandwidth = EvaluatePeakBandwidth(ind, false);
        ind.peak_nip_bandwidth = EvaluatePeakBandwidth(ind, true);
        if (config_.moham.enforce_bandwidth_constraints)
        {
            constexpr double tolerance = 1e-8;
            if (config_.moham.system_bandwidth > 0.0 &&
                ind.peak_system_bandwidth >
                    config_.moham.system_bandwidth * (1.0 + tolerance))
                throw std::runtime_error(
                    "resource scheduler exceeded the system bandwidth limit");
            if (config_.moham.nip_link_bandwidth > 0.0 &&
                ind.peak_nip_bandwidth >
                    config_.moham.nip_link_bandwidth * (1.0 + tolerance))
                throw std::runtime_error(
                    "resource scheduler exceeded a NoP link bandwidth limit");
        }
        ind.peak_temperature = config_.moham.thermal_enabled
            ? EvaluatePeakTemperature(ind) : 0.0;
    }


    double Moham::EvaluatePeakTemperature(const Individual& ind) const
    {
        const std::size_t num_chiplets = ind.accelerators.size();
        if (num_chiplets == 0) return config_.moham.thermal_ambient_temperature;

        std::vector<double> boundaries;
        boundaries.reserve(ind.layer_evals.size() * 2);
        for (const auto& evaluation : ind.layer_evals)
        {
            boundaries.push_back(evaluation.start_time);
            boundaries.push_back(evaluation.end_time);
        }
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());

        std::vector<double> temperatures(num_chiplets, config_.moham.thermal_ambient_temperature);
        double package_temperature =
            config_.moham.thermal_ambient_temperature;
        double peak_temperature = config_.moham.thermal_ambient_temperature;
        for (std::size_t interval = 0; interval + 1 < boundaries.size(); ++interval)
        {
            const double start = boundaries[interval];
            const double end = boundaries[interval + 1];
            if (end <= start) continue;

            std::vector<double> powers(num_chiplets, 0.0);
            for (Layer::ID layer_id = 0; layer_id < graph_->NumLayers(); ++layer_id)
            {
                const auto& evaluation = ind.layer_evals[layer_id];
                if (evaluation.start_time <= start && end <= evaluation.end_time)
                {
                    const double execution_cycles = std::max(evaluation.end_time - evaluation.start_time, 1e-12);
                    const double interval_energy =
                        evaluation.energy + EvaluateLayerCommunicationEnergy(ind, layer_id);
                    powers[ind.genome[layer_id].acc_instance_id] += interval_energy / execution_cycles *
                        config_.moham.thermal_frequency_ghz * 1e-3;
                }
            }

            const double interval_cycles = end - start;
            const std::size_t steps = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(
                interval_cycles / config_.moham.thermal_time_step_cycles)));
            const double dt_seconds = interval_cycles /
                (config_.moham.thermal_frequency_ghz * 1e9 * steps);

            for (std::size_t step = 0; step < steps; ++step)
            {
                std::vector<double> next_temperatures = temperatures;
                double heat_to_package = 0.0;
                for (ArchInstance::ID acc_id = 0; acc_id < num_chiplets; ++acc_id)
                {
                    // Timeloop reports area in um^2; the RC parameters are
                    // area-normalized in mm^2.
                    const double area_mm2 = std::max(ind.accelerators[acc_id].area * 1e-6, 1e-12);
                    const double capacitance = config_.moham.thermal_capacitance_per_area * area_mm2;
                    const double self_conductance = area_mm2 / config_.moham.thermal_self_resistance;
                    const double chiplet_to_package = self_conductance *
                        (temperatures[acc_id] - package_temperature);
                    double heat_out = chiplet_to_package;
                    heat_to_package += chiplet_to_package;

                    for (ArchInstance::ID other_id = 0; other_id < num_chiplets; ++other_id)
                    {
                        if (other_id == acc_id) continue;
                        const auto lhs = ind.accelerators[acc_id].mesh_coordinates;
                        const auto rhs = ind.accelerators[other_id].mesh_coordinates;
                        const int distance = std::abs(lhs.first - rhs.first) + std::abs(lhs.second - rhs.second);
                        if (distance > 0)
                            heat_out += (temperatures[acc_id] - temperatures[other_id]) /
                                (config_.moham.thermal_neighbor_resistance * distance);
                    }
                    next_temperatures[acc_id] += dt_seconds * (powers[acc_id] - heat_out) / capacitance;
                    peak_temperature = std::max(peak_temperature, next_temperatures[acc_id]);
                }
                const double package_to_ambient =
                    (package_temperature -
                     config_.moham.thermal_ambient_temperature) /
                    config_.moham.thermal_package_resistance;
                package_temperature +=
                    dt_seconds *
                    (heat_to_package - package_to_ambient) /
                    config_.moham.thermal_package_capacitance;
                temperatures.swap(next_temperatures);
            }
        }
        return peak_temperature;
    }


    void Moham::EvaluateConstraints(Individual& ind)
    {
        ind.constraint_profile.fill(0.0);
        const auto set_upper_bound = [&ind](ConstraintMetric metric, double value, double limit) {
            if (limit > 0.0 && value > limit)
                ind.constraint_profile[metric] = std::max(ind.constraint_profile[metric], (value - limit) / limit);
        };

        set_upper_bound(PACKAGE_POWER_VIOLATION, ind.peak_package_power, config_.moham.package_power_max);
        for (double chiplet_power : ind.peak_chiplet_power)
            set_upper_bound(CHIPLET_POWER_VIOLATION, chiplet_power, config_.moham.chiplet_power_max);
        set_upper_bound(AREA_VIOLATION, ind.objectives[2], config_.moham.package_area_max);

        if (config_.moham.thermal_enabled)
            set_upper_bound(
                TEMPERATURE_VIOLATION,
                ind.peak_temperature +
                    config_.moham.thermal_validation_margin,
                config_.moham.thermal_temperature_max);

        if (config_.moham.enforce_bandwidth_constraints)
        {
            set_upper_bound(BANDWIDTH_VIOLATION, ind.peak_system_bandwidth, config_.moham.system_bandwidth);
            set_upper_bound(BANDWIDTH_VIOLATION, ind.peak_nip_bandwidth, config_.moham.nip_link_bandwidth);
        }

        for (std::size_t network_id = 0;
             network_id < ind.network_completion_times.size() && network_id < config_.moham.workload_deadlines.size();
             ++network_id)
            set_upper_bound(SLO_VIOLATION, ind.network_completion_times[network_id], config_.moham.workload_deadlines[network_id]);

        ind.constraint_violation = std::accumulate(
            ind.constraint_profile.begin(), ind.constraint_profile.end(), 0.0);
    }


    void Moham::BuildBottleneckLedger(Individual& ind) const
    {
        ind.bottleneck_ledger.clear();
        const auto add = [&ind](BottleneckKind kind, Layer::ID layer_id,
                                ArchInstance::ID accelerator_id, double contribution) {
            if (contribution > 0.0)
                ind.bottleneck_ledger.push_back({kind, layer_id, accelerator_id, contribution});
        };

        // SLO attribution is the final layer of the late workload under the
        // decoded DAG schedule. This remains meaningful for a batch workload.
        for (std::size_t network_id = 0;
             network_id < ind.network_completion_times.size() && network_id < config_.moham.workload_deadlines.size();
             ++network_id)
        {
            const double deadline = config_.moham.workload_deadlines[network_id];
            if (deadline <= 0.0 || ind.network_completion_times[network_id] <= deadline) continue;
            Layer::ID tail_layer = 0;
            double tail_time = -1.0;
            for (Layer::ID layer_id = 0; layer_id < graph_->NumLayers(); ++layer_id)
                if (layer_network_ids_[layer_id] == network_id && ind.layer_evals[layer_id].end_time > tail_time)
                {
                    tail_layer = layer_id;
                    tail_time = ind.layer_evals[layer_id].end_time;
                }
            add(BottleneckKind::SLO, tail_layer, ind.genome[tail_layer].acc_instance_id,
                (ind.network_completion_times[network_id] - deadline) / deadline);
        }

        ArchInstance::ID hottest_accelerator = 0;
        for (ArchInstance::ID acc_id = 1; acc_id < ind.peak_chiplet_power.size(); ++acc_id)
            if (ind.peak_chiplet_power[acc_id] > ind.peak_chiplet_power[hottest_accelerator])
                hottest_accelerator = acc_id;

        Layer::ID hottest_layer = 0;
        double hottest_layer_power = -1.0;
        for (Layer::ID layer_id = 0; layer_id < graph_->NumLayers(); ++layer_id)
            if (ind.genome[layer_id].acc_instance_id == hottest_accelerator)
            {
                const auto& evaluation = ind.layer_evals[layer_id];
                const double power_proxy = evaluation.energy / std::max(evaluation.cycles, 1e-12);
                if (power_proxy > hottest_layer_power)
                {
                    hottest_layer = layer_id;
                    hottest_layer_power = power_proxy;
                }
            }

        add(BottleneckKind::PACKAGE_POWER, hottest_layer, hottest_accelerator,
            ind.constraint_profile[PACKAGE_POWER_VIOLATION]);
        add(BottleneckKind::CHIPLET_POWER, hottest_layer, hottest_accelerator,
            ind.constraint_profile[CHIPLET_POWER_VIOLATION]);
        add(BottleneckKind::TEMPERATURE, hottest_layer, hottest_accelerator,
            ind.constraint_profile[TEMPERATURE_VIOLATION]);

        Layer::ID bandwidth_layer = 0;
        ArchInstance::ID bandwidth_accelerator = 0;
        double bandwidth_contribution = ind.constraint_profile[BANDWIDTH_VIOLATION];
        const auto inspect_delay = [&](const BottleneckSegment& segment) {
            const double delay = std::max(0.0, segment.end_time - segment.start_time);
            const double makespan = std::max(ind.objectives[1], 1.0);
            if (delay / makespan > bandwidth_contribution)
            {
                bandwidth_contribution = delay / makespan;
                bandwidth_layer = segment.layer_id;
                bandwidth_accelerator = segment.acc;
            }
        };
        for (const auto& segment : ind.nip_bottlenecks) inspect_delay(segment);
        for (const auto& segment : ind.memory_bottlenecks) inspect_delay(segment);
        if (config_.moham.bandwidth_service_aware_repair)
        {
            constexpr double epsilon = 1e-9;
            const double makespan = std::max(ind.objectives[1], 1.0);
            for (Layer::ID layer_id = 0; layer_id < graph_->NumLayers(); ++layer_id)
            {
                const auto& evaluation = ind.layer_evals[layer_id];
                const double scheduled_duration = std::max(
                    1.0, evaluation.end_time - evaluation.start_time);
                const double compute_only_duration = std::max(
                    1.0, std::ceil(evaluation.cycles - epsilon));
                const double service_stretch = std::max(
                    0.0, scheduled_duration - compute_only_duration);
                if (service_stretch / makespan > bandwidth_contribution)
                {
                    bandwidth_contribution = service_stretch / makespan;
                    bandwidth_layer = layer_id;
                    bandwidth_accelerator = ind.genome[layer_id].acc_instance_id;
                }
            }
        }
        add(BottleneckKind::BANDWIDTH, bandwidth_layer, bandwidth_accelerator,
            bandwidth_contribution);

        if (!ind.genome.empty())
            add(BottleneckKind::AREA, 0, ind.genome[0].acc_instance_id,
                ind.constraint_profile[AREA_VIOLATION]);

        std::sort(ind.bottleneck_ledger.begin(), ind.bottleneck_ledger.end(),
                  [](const BottleneckAttribution& lhs, const BottleneckAttribution& rhs) {
                      return lhs.contribution > rhs.contribution;
                  });
    }


    bool Moham::ConstraintProfileDominates(const Individual& a, const Individual& b) const
    {
        constexpr double epsilon = 1e-12;
        bool strictly_better = false;
        for (std::size_t metric = 0; metric < CONSTRAINT_METRIC_COUNT; ++metric)
        {
            if (a.constraint_profile[metric] > b.constraint_profile[metric] + epsilon)
                return false;
            if (a.constraint_profile[metric] + epsilon < b.constraint_profile[metric])
                strictly_better = true;
        }
        return strictly_better;
    }


    double Moham::ConstraintProfileDistance(const Individual& a, const Individual& b) const
    {
        double squared_distance = 0.0;
        for (std::size_t metric = 0; metric < CONSTRAINT_METRIC_COUNT; ++metric)
        {
            const double difference = a.constraint_profile[metric] - b.constraint_profile[metric];
            squared_distance += difference * difference;
        }
        return std::sqrt(squared_distance);
    }


    void Moham::UpdateSCBAArchives(const Population& candidates)
    {
        const std::size_t archive_size = config_.moham.boundary_archive_size
            ? config_.moham.boundary_archive_size : std::max(1u, config_.moham.population_size / 3);

        Population feasible_candidates = feasible_archive_;
        Population infeasible_candidates = boundary_archive_;
        for (const auto& individual : candidates)
            if (individual.constraint_violation <= 1e-12)
                feasible_candidates.push_back(individual);
            else
                infeasible_candidates.push_back(individual);

        if (feasible_candidates.size() <= archive_size)
        {
            feasible_archive_ = std::move(feasible_candidates);
            if (!feasible_archive_.empty()) AssignRankAndCrowdingDistance(feasible_archive_);
        }
        else
        {
            AssignRankAndCrowdingDistance(feasible_candidates);
            Population next_archive(archive_size);
            Survival(next_archive, feasible_candidates, false, rng_[0]);
            feasible_archive_ = std::move(next_archive);
        }

        if (config_.moham.boundary_archive_mode == "scalar")
        {
            std::stable_sort(
                infeasible_candidates.begin(), infeasible_candidates.end(),
                [](const Individual& lhs, const Individual& rhs) {
                    constexpr double epsilon = 1e-12;
                    if (lhs.constraint_violation + epsilon <
                        rhs.constraint_violation)
                        return true;
                    if (rhs.constraint_violation + epsilon <
                        lhs.constraint_violation)
                        return false;
                    if (lhs.constraint_profile != rhs.constraint_profile)
                        return lhs.constraint_profile < rhs.constraint_profile;
                    return lhs.design_fingerprint < rhs.design_fingerprint;
                });
            if (infeasible_candidates.size() > archive_size)
                infeasible_candidates.resize(archive_size);
            boundary_archive_ = std::move(infeasible_candidates);
            return;
        }

        Population profile_front;
        profile_front.reserve(infeasible_candidates.size());
        for (std::size_t candidate_id = 0; candidate_id < infeasible_candidates.size(); ++candidate_id)
        {
            bool dominated = false;
            for (std::size_t other_id = 0; other_id < infeasible_candidates.size(); ++other_id)
                if (candidate_id != other_id &&
                    ConstraintProfileDominates(infeasible_candidates[other_id], infeasible_candidates[candidate_id]))
                {
                    dominated = true;
                    break;
                }
            if (!dominated) profile_front.push_back(infeasible_candidates[candidate_id]);
        }

        if (profile_front.size() <= archive_size)
        {
            boundary_archive_ = std::move(profile_front);
            return;
        }

        Population next_boundary;
        next_boundary.reserve(archive_size);
        std::vector<bool> selected(profile_front.size(), false);
        const auto best_violation = std::min_element(profile_front.begin(), profile_front.end(),
            [](const Individual& lhs, const Individual& rhs) { return lhs.constraint_violation < rhs.constraint_violation; });
        selected[std::distance(profile_front.begin(), best_violation)] = true;
        next_boundary.push_back(*best_violation);

        while (next_boundary.size() < archive_size)
        {
            std::size_t best_id = profile_front.size();
            double best_distance = -1.0;
            for (std::size_t candidate_id = 0; candidate_id < profile_front.size(); ++candidate_id)
            {
                if (selected[candidate_id]) continue;
                double nearest_distance = std::numeric_limits<double>::max();
                for (const auto& selected_individual : next_boundary)
                    nearest_distance = std::min(nearest_distance,
                                                ConstraintProfileDistance(profile_front[candidate_id], selected_individual));
                if (nearest_distance > best_distance + 1e-12 ||
                    (std::abs(nearest_distance - best_distance) <= 1e-12 &&
                     (best_id == profile_front.size() ||
                      profile_front[candidate_id].constraint_violation < profile_front[best_id].constraint_violation)))
                {
                    best_distance = nearest_distance;
                    best_id = candidate_id;
                }
            }
            if (best_id == profile_front.size()) break;
            selected[best_id] = true;
            next_boundary.push_back(profile_front[best_id]);
        }
        boundary_archive_ = std::move(next_boundary);
    }


    void Moham::UpdateReferenceDirectionBias()
    {
        const std::size_t archive_capacity = config_.moham.boundary_archive_size
            ? config_.moham.boundary_archive_size
            : std::max(1u, config_.moham.population_size / 3);
        const double feasible_saturation = std::min(
            1.0, static_cast<double>(feasible_archive_.size()) / archive_capacity);
        // Constraint-boundary pressure is useful while feasible solutions are
        // rare, but the ablation showed that keeping it active after the
        // feasible archive fills damages objective extremes. Fade it
        // quadratically and recover the original simplex directions once the
        // feasible archive reaches capacity.
        const double effective_adaptation_rate =
            config_.moham.reference_adaptation_rate *
            std::pow(std::max(0.0, 1.0 - feasible_saturation), 2.0);

        std::vector<double> bias(5, 1.0);
        if (effective_adaptation_rate <= 1e-12)
        {
            ConfigureReferenceDirectionBias(std::move(bias));
            return;
        }

        std::array<double, 5> pressure{};
        for (const auto& individual : boundary_archive_)
        {
            pressure[0] += individual.constraint_profile[PACKAGE_POWER_VIOLATION] +
                           individual.constraint_profile[CHIPLET_POWER_VIOLATION];
            pressure[1] += individual.constraint_profile[SLO_VIOLATION] +
                           individual.constraint_profile[BANDWIDTH_VIOLATION];
            pressure[2] += individual.constraint_profile[AREA_VIOLATION];
            pressure[3] += individual.constraint_profile[TEMPERATURE_VIOLATION] +
                           individual.constraint_profile[PACKAGE_POWER_VIOLATION];
            pressure[4] += individual.constraint_profile[SLO_VIOLATION];
        }

        const double scale = std::accumulate(pressure.begin(), pressure.end(), 0.0);
        if (scale > 0.0)
            for (std::size_t objective = 0; objective < bias.size(); ++objective)
                bias[objective] += effective_adaptation_rate * 5.0 *
                    pressure[objective] / scale;
        ConfigureReferenceDirectionBias(std::move(bias));
    }


    void Moham::UpdateSCBABoundaryPolicy()
    {
        const std::size_t archive_capacity = config_.moham.boundary_archive_size
            ? config_.moham.boundary_archive_size
            : std::max(1u, config_.moham.population_size / 3);
        const double feasible_archive_saturation = std::min(
            1.0, static_cast<double>(feasible_archive_.size()) /
                archive_capacity);

        if (!config_.moham.adaptive_boundary_parenting)
        {
            effective_boundary_parent_probability_ =
                config_.moham.boundary_parent_probability *
                std::pow(std::max(0.0, 1.0 - feasible_archive_saturation), 4.0);
            return;
        }

        if (feasible_archive_.empty())
        {
            // Until the first feasible design is found, the diverse boundary
            // archive is the only constraint-aware parent source.
            effective_boundary_parent_probability_ = 1.0;
            return;
        }

        const auto feasible_parent_count = std::count_if(
            parent_population_.begin(), parent_population_.end(),
            [](const Individual& individual) {
                return individual.constraint_violation <= 1e-12;
            });
        const double feasible_parent_fraction = parent_population_.empty()
            ? 0.0
            : static_cast<double>(feasible_parent_count) /
                  parent_population_.size();
        const double feasible_scarcity =
            std::max(0.0, 1.0 - feasible_parent_fraction);

        std::array<double, CONSTRAINT_METRIC_COUNT> constraint_mass{};
        double total_constraint_mass = 0.0;
        for (const auto& individual : boundary_archive_)
            for (std::size_t metric = 0; metric < CONSTRAINT_METRIC_COUNT; ++metric)
            {
                constraint_mass[metric] += individual.constraint_profile[metric];
                total_constraint_mass += individual.constraint_profile[metric];
            }

        double normalized_entropy = 0.0;
        if (total_constraint_mass > 1e-12)
        {
            for (double mass : constraint_mass)
                if (mass > 1e-12)
                {
                    const double share = mass / total_constraint_mass;
                    normalized_entropy -= share * std::log(share);
                }
            normalized_entropy /= std::log(
                static_cast<double>(CONSTRAINT_METRIC_COUNT));
        }
        normalized_entropy = std::max(0.0, std::min(1.0, normalized_entropy));

        const double mean_boundary_violation = boundary_archive_.empty()
            ? 0.0
            : total_constraint_mass / boundary_archive_.size();
        const double boundary_proximity =
            1.0 / (1.0 + std::max(0.0, mean_boundary_violation));
        const double profile_diversity_weight =
            0.5 + 0.5 * normalized_entropy;

        effective_boundary_parent_probability_ = std::max(
            0.0, std::min(
                config_.moham.boundary_parent_probability,
                config_.moham.boundary_parent_probability *
                    feasible_scarcity * profile_diversity_weight *
                    boundary_proximity));
    }


    const Moham::Individual& Moham::SelectSCBAParent()
    {
        auto& rng = rng_[omp_get_thread_num()];
        scba_parent_selection_count_.fetch_add(1, std::memory_order_relaxed);
        scba_effective_probability_sum_ppm_.fetch_add(
            static_cast<unsigned long long>(std::llround(
                1.0e6 * effective_boundary_parent_probability_)),
            std::memory_order_relaxed);
        // The fixed CPAR configuration deliberately caps boundary-parent
        // sampling.  Only the explicitly adaptive policy is allowed to make
        // the boundary archive the sole parent source before its first
        // feasible design.  Otherwise an empty feasible archive must retain
        // ordinary-population exploration; forcing every parent through the
        // boundary archive would silently override the configured cap.
        const bool forced_boundary = config_.moham.adaptive_boundary_parenting &&
            feasible_archive_.empty();
        const bool select_boundary = !boundary_archive_.empty() &&
            (forced_boundary ||
             uni_distribution_(rng) < effective_boundary_parent_probability_);
        if (select_boundary)
        {
            scba_boundary_parent_selection_count_.fetch_add(
                1, std::memory_order_relaxed);
            if (forced_boundary)
                scba_forced_boundary_parent_selection_count_.fetch_add(
                    1, std::memory_order_relaxed);
            const Individual& first = boundary_archive_[rng() % boundary_archive_.size()];
            const Individual& second = boundary_archive_[rng() % boundary_archive_.size()];
            if (first.constraint_violation < second.constraint_violation) return first;
            if (second.constraint_violation < first.constraint_violation) return second;
            return (rng() & 1u) ? first : second;
        }
        if (!feasible_archive_.empty()) return feasible_archive_[Tournament(feasible_archive_)];
        return parent_population_[Tournament(parent_population_)];
    }


    std::vector<std::size_t> Moham::AssociateCTAEARegions(
        const Population& population,
        const std::vector<std::vector<double>>& directions) const
    {
        std::vector<std::size_t> regions(population.size(), 0);
        if (population.empty() || directions.empty()) return regions;
        constexpr std::size_t objectives = 5;
        std::array<double, objectives> ideal;
        std::array<double, objectives> nadir;
        ideal.fill(std::numeric_limits<double>::max());
        nadir.fill(std::numeric_limits<double>::lowest());
        for (const auto& individual : population)
            for (std::size_t objective = 0; objective < objectives; ++objective)
            {
                ideal[objective] = std::min(
                    ideal[objective], individual.objectives[objective]);
                nadir[objective] = std::max(
                    nadir[objective], individual.objectives[objective]);
            }

        for (std::size_t individual_id = 0;
             individual_id < population.size(); ++individual_id)
        {
            std::array<double, objectives> normalised{};
            double normalised_norm = 0.0;
            for (std::size_t objective = 0; objective < objectives; ++objective)
            {
                const double range = std::max(
                    nadir[objective] - ideal[objective], 1e-12);
                normalised[objective] =
                    (population[individual_id].objectives[objective] -
                     ideal[objective]) / range;
                normalised_norm +=
                    normalised[objective] * normalised[objective];
            }
            normalised_norm = std::sqrt(normalised_norm);

            double best_cosine = std::numeric_limits<double>::lowest();
            for (std::size_t direction_id = 0;
                 direction_id < directions.size(); ++direction_id)
            {
                double dot = 0.0;
                double direction_norm = 0.0;
                for (std::size_t objective = 0; objective < objectives; ++objective)
                {
                    dot += normalised[objective] *
                        directions[direction_id][objective];
                    direction_norm +=
                        directions[direction_id][objective] *
                        directions[direction_id][objective];
                }
                const double cosine =
                    normalised_norm > 1e-12 && direction_norm > 1e-12
                    ? dot / (normalised_norm * std::sqrt(direction_norm))
                    : (direction_id == 0 ? 1.0 : 0.0);
                if (cosine > best_cosine)
                {
                    best_cosine = cosine;
                    regions[individual_id] = direction_id;
                }
            }
        }
        return regions;
    }


    Moham::Population Moham::UpdateCTAEAConvergence(
        const Population& candidates,
        const std::vector<std::vector<double>>& directions)
    {
        const std::size_t target = config_.moham.population_size;
        Population combined = ctaea_convergence_archive_;
        combined.insert(combined.end(), candidates.begin(), candidates.end());
        Population feasible;
        Population infeasible;
        for (const auto& individual : combined)
            (individual.constraint_violation <= 1e-12
                ? feasible : infeasible).push_back(individual);

        Population selected;
        if (feasible.size() >= target)
        {
            AssignRankAndCrowdingDistance(feasible);
            std::vector<std::size_t> order(feasible.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                [&](std::size_t lhs, std::size_t rhs) {
                    return feasible[lhs].rank < feasible[rhs].rank;
                });
            const unsigned splitting_rank = feasible[order[target - 1]].rank;
            for (const auto& individual : feasible)
                if (individual.rank <= splitting_rank)
                    selected.push_back(individual);

            while (selected.size() > target)
            {
                const auto regions = AssociateCTAEARegions(selected, directions);
                std::vector<std::size_t> counts(directions.size(), 0);
                for (std::size_t region : regions) counts[region]++;
                const std::size_t crowded_region = std::distance(
                    counts.begin(), std::max_element(counts.begin(), counts.end()));
                std::vector<std::size_t> crowded;
                for (std::size_t id = 0; id < selected.size(); ++id)
                    if (regions[id] == crowded_region) crowded.push_back(id);

                std::size_t remove_id = crowded.front();
                if (crowded.size() > 1)
                {
                    std::size_t first = crowded[0], second = crowded[1];
                    double nearest = std::numeric_limits<double>::max();
                    std::array<double, 5> ideal;
                    std::array<double, 5> nadir;
                    ideal.fill(std::numeric_limits<double>::max());
                    nadir.fill(std::numeric_limits<double>::lowest());
                    for (const auto& individual : selected)
                        for (std::size_t objective = 0; objective < 5; ++objective)
                        {
                            ideal[objective] = std::min(
                                ideal[objective], individual.objectives[objective]);
                            nadir[objective] = std::max(
                                nadir[objective], individual.objectives[objective]);
                        }
                    const auto normalised_value = [&](std::size_t id, std::size_t objective) {
                        return (selected[id].objectives[objective] - ideal[objective]) /
                            std::max(nadir[objective] - ideal[objective], 1e-12);
                    };
                    for (std::size_t left = 0; left < crowded.size(); ++left)
                        for (std::size_t right = left + 1; right < crowded.size(); ++right)
                        {
                            double distance = 0.0;
                            for (std::size_t objective = 0; objective < 5; ++objective)
                            {
                                const double difference =
                                    normalised_value(crowded[left], objective) -
                                    normalised_value(crowded[right], objective);
                                distance += difference * difference;
                            }
                            if (distance < nearest)
                            {
                                nearest = distance;
                                first = crowded[left];
                                second = crowded[right];
                            }
                        }
                    const auto scalarisation = [&](std::size_t id) {
                        double value = 0.0;
                        for (std::size_t objective = 0; objective < 5; ++objective)
                            value = std::max(
                                value,
                                normalised_value(id, objective) /
                                std::max(
                                    directions[regions[id]][objective], 1e-6));
                        return value;
                    };
                    remove_id = scalarisation(first) > scalarisation(second)
                        ? first : second;
                }
                selected.erase(selected.begin() + remove_id);
            }
        }
        else
        {
            selected = feasible;
            if (!infeasible.empty() && selected.size() < target)
            {
                const auto regions =
                    AssociateCTAEARegions(infeasible, directions);
                std::array<double, 5> ideal;
                std::array<double, 5> nadir;
                ideal.fill(std::numeric_limits<double>::max());
                nadir.fill(std::numeric_limits<double>::lowest());
                for (const auto& individual : infeasible)
                    for (std::size_t objective = 0; objective < 5; ++objective)
                    {
                        ideal[objective] = std::min(
                            ideal[objective], individual.objectives[objective]);
                        nadir[objective] = std::max(
                            nadir[objective], individual.objectives[objective]);
                    }
                std::vector<std::pair<double, double>> metrics(infeasible.size());
                for (std::size_t id = 0; id < infeasible.size(); ++id)
                {
                    double scalarisation = 0.0;
                    for (std::size_t objective = 0; objective < 5; ++objective)
                    {
                        const double normalised =
                            (infeasible[id].objectives[objective] - ideal[objective]) /
                            std::max(nadir[objective] - ideal[objective], 1e-12);
                        scalarisation = std::max(
                            scalarisation,
                            normalised /
                            std::max(directions[regions[id]][objective], 1e-6));
                    }
                    metrics[id] = {
                        infeasible[id].constraint_violation, scalarisation};
                }

                std::vector<unsigned> ranks(infeasible.size(), 0);
                std::vector<bool> remaining(infeasible.size(), true);
                unsigned rank = 0;
                std::size_t remaining_count = infeasible.size();
                while (remaining_count)
                {
                    std::vector<std::size_t> front;
                    for (std::size_t id = 0; id < infeasible.size(); ++id)
                    {
                        if (!remaining[id]) continue;
                        bool dominated = false;
                        for (std::size_t other = 0; other < infeasible.size(); ++other)
                        {
                            if (id == other || !remaining[other]) continue;
                            const bool no_worse =
                                metrics[other].first <= metrics[id].first &&
                                metrics[other].second <= metrics[id].second;
                            const bool better =
                                metrics[other].first < metrics[id].first ||
                                metrics[other].second < metrics[id].second;
                            if (no_worse && better)
                            {
                                dominated = true;
                                break;
                            }
                        }
                        if (!dominated) front.push_back(id);
                    }
                    for (std::size_t id : front)
                    {
                        remaining[id] = false;
                        ranks[id] = rank;
                    }
                    remaining_count -= front.size();
                    ++rank;
                }
                std::vector<std::size_t> order(infeasible.size());
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                    [&](std::size_t lhs, std::size_t rhs) {
                        if (ranks[lhs] != ranks[rhs])
                            return ranks[lhs] < ranks[rhs];
                        return infeasible[lhs].constraint_violation <
                            infeasible[rhs].constraint_violation;
                    });
                for (std::size_t id : order)
                    if (selected.size() < target)
                        selected.push_back(infeasible[id]);
            }
        }
        AssignRankAndCrowdingDistance(selected);
        return selected;
    }


    Moham::Population Moham::UpdateCTAEADiversity(
        const Population& candidates,
        const std::vector<std::vector<double>>& directions)
    {
        const std::size_t target = config_.moham.population_size;
        Population pool = ctaea_diversity_archive_;
        pool.insert(pool.end(), candidates.begin(), candidates.end());
        const auto pool_regions = AssociateCTAEARegions(pool, directions);
        const auto convergence_regions =
            AssociateCTAEARegions(ctaea_convergence_archive_, directions);
        std::vector<std::size_t> convergence_counts(directions.size(), 0);
        for (std::size_t region : convergence_regions)
            convergence_counts[region]++;

        std::vector<bool> selected_flag(pool.size(), false);
        Population selected;
        selected.reserve(target);
        for (std::size_t iteration = 1;
             selected.size() < target && iteration <= target + 1; ++iteration)
        {
            bool progress = false;
            for (std::size_t region = 0;
                 region < directions.size() && selected.size() < target; ++region)
            {
                const std::size_t required = iteration > convergence_counts[region]
                    ? iteration - convergence_counts[region] : 0;
                for (std::size_t pick = 0;
                     pick < required && selected.size() < target; ++pick)
                {
                    std::vector<std::size_t> available;
                    for (std::size_t id = 0; id < pool.size(); ++id)
                        if (!selected_flag[id] && pool_regions[id] == region)
                            available.push_back(id);
                    if (available.empty()) break;

                    std::vector<std::size_t> nondominated;
                    for (std::size_t id : available)
                    {
                        bool dominated = false;
                        for (std::size_t other : available)
                        {
                            if (id == other) continue;
                            bool no_worse = true, strictly_better = false;
                            for (std::size_t objective = 0; objective < 5; ++objective)
                            {
                                no_worse = no_worse &&
                                    pool[other].objectives[objective] <=
                                    pool[id].objectives[objective];
                                strictly_better = strictly_better ||
                                    pool[other].objectives[objective] <
                                    pool[id].objectives[objective];
                            }
                            if (no_worse && strictly_better)
                            {
                                dominated = true;
                                break;
                            }
                        }
                        if (!dominated) nondominated.push_back(id);
                    }

                    std::array<double, 5> ideal;
                    std::array<double, 5> nadir;
                    ideal.fill(std::numeric_limits<double>::max());
                    nadir.fill(std::numeric_limits<double>::lowest());
                    for (std::size_t id : nondominated)
                        for (std::size_t objective = 0; objective < 5; ++objective)
                        {
                            ideal[objective] = std::min(
                                ideal[objective], pool[id].objectives[objective]);
                            nadir[objective] = std::max(
                                nadir[objective], pool[id].objectives[objective]);
                        }
                    std::size_t best = nondominated.front();
                    double best_value = std::numeric_limits<double>::max();
                    for (std::size_t id : nondominated)
                    {
                        double scalarisation = 0.0;
                        for (std::size_t objective = 0; objective < 5; ++objective)
                        {
                            const double normalised =
                                (pool[id].objectives[objective] - ideal[objective]) /
                                std::max(nadir[objective] - ideal[objective], 1e-12);
                            scalarisation = std::max(
                                scalarisation,
                                normalised /
                                std::max(directions[region][objective], 1e-6));
                        }
                        if (scalarisation < best_value)
                        {
                            best_value = scalarisation;
                            best = id;
                        }
                    }
                    selected_flag[best] = true;
                    selected.push_back(pool[best]);
                    progress = true;
                }
            }
            if (!progress && iteration > target) break;
        }

        if (selected.size() < target)
        {
            Population fallback;
            for (std::size_t id = 0; id < pool.size(); ++id)
                if (!selected_flag[id]) fallback.push_back(pool[id]);
            AssignRankAndCrowdingDistance(fallback);
            std::stable_sort(fallback.begin(), fallback.end(),
                [](const Individual& lhs, const Individual& rhs) {
                    if (lhs.rank != rhs.rank) return lhs.rank < rhs.rank;
                    return lhs.constraint_violation < rhs.constraint_violation;
                });
            for (const auto& individual : fallback)
                if (selected.size() < target)
                    selected.push_back(individual);
        }
        AssignRankAndCrowdingDistance(selected);
        return selected;
    }


    void Moham::UpdateCTAEAArchives(const Population& candidates)
    {
        unsigned partitions = config_.moham.reference_partitions;
        if (partitions == 0)
        {
            partitions = 1;
            while (ReferenceDirectionCount(5, partitions + 1) <=
                   config_.moham.population_size)
                ++partitions;
        }
        const auto directions = BuildReferenceDirections(5, partitions);
        ctaea_convergence_archive_ =
            UpdateCTAEAConvergence(candidates, directions);
        ctaea_diversity_archive_ =
            UpdateCTAEADiversity(candidates, directions);
    }


    void Moham::UpdateCTAEAMatingProbabilities()
    {
        Population combined = ctaea_convergence_archive_;
        combined.insert(
            combined.end(), ctaea_diversity_archive_.begin(),
            ctaea_diversity_archive_.end());
        AssignRankAndCrowdingDistance(combined);
        std::size_t convergence_nondominated = 0;
        std::size_t diversity_nondominated = 0;
        for (std::size_t id = 0; id < combined.size(); ++id)
            if (combined[id].rank == 0)
                (id < ctaea_convergence_archive_.size()
                    ? convergence_nondominated
                    : diversity_nondominated)++;
        ctaea_first_parent_from_convergence_ =
            convergence_nondominated > diversity_nondominated ? 1.0 : 0.0;

        Population convergence = ctaea_convergence_archive_;
        AssignRankAndCrowdingDistance(convergence);
        const std::size_t nondominated = std::count_if(
            convergence.begin(), convergence.end(),
            [](const Individual& individual) { return individual.rank == 0; });
        ctaea_second_parent_convergence_probability_ =
            convergence.empty() ? 0.0
            : static_cast<double>(nondominated) / convergence.size();
    }


    const Moham::Individual& Moham::CTAEAMatingSelection(
        const Population& archive)
    {
        auto& rng = rng_[omp_get_thread_num()];
        const Individual& first = archive[rng() % archive.size()];
        const Individual& second = archive[rng() % archive.size()];
        const bool first_feasible = first.constraint_violation <= 1e-12;
        const bool second_feasible = second.constraint_violation <= 1e-12;
        if (!first_feasible && !second_feasible)
            return archive[rng() % archive.size()];
        if (first_feasible != second_feasible)
            return first_feasible ? first : second;

        bool second_no_worse = true;
        bool second_strictly_better = false;
        for (std::size_t objective = 0; objective < 5; ++objective)
        {
            second_no_worse = second_no_worse &&
                second.objectives[objective] <= first.objectives[objective];
            second_strictly_better = second_strictly_better ||
                second.objectives[objective] < first.objectives[objective];
        }
        return second_no_worse && second_strictly_better ? second : first;
    }


    const Moham::Individual& Moham::SelectCTAEAParent(bool first_parent)
    {
        auto& rng = rng_[omp_get_thread_num()];
        const bool from_convergence = first_parent
            ? ctaea_first_parent_from_convergence_ > 0.5
            : uni_distribution_(rng) <
                ctaea_second_parent_convergence_probability_;
        const Population& preferred = from_convergence
            ? ctaea_convergence_archive_ : ctaea_diversity_archive_;
        const Population& fallback = from_convergence
            ? ctaea_diversity_archive_ : ctaea_convergence_archive_;
        return CTAEAMatingSelection(
            preferred.empty() ? fallback : preferred);
    }


    bool Moham::MoveLayerToCompatibleAccelerator(Individual& ind, Layer::ID layer_id, bool prefer_coolest) const
    {
        if (layer_id >= ind.genome.size() || ind.accelerators.size() < 2) return false;
        const ArchInstance::ID source_id = ind.genome[layer_id].acc_instance_id;
        const Workload::ID workload_id = (*graph_)[layer_id].workload_id;
        ArchInstance::ID best_destination = source_id;
        Medea::Individual::ID best_mapping = ind.genome[layer_id].mapping_id;
        double best_score = std::numeric_limits<double>::max();
        for (ArchInstance::ID destination_id = 0; destination_id < ind.accelerators.size(); ++destination_id)
        {
            if (destination_id == source_id) continue;
            const ArchTemplate::ID template_id = ind.accelerators[destination_id].arch_template_id;
            const auto mappings = workload_mappings_[workload_id].find(template_id);
            if (mappings == workload_mappings_[workload_id].end() || mappings->second.empty()) continue;
            Medea::Individual::ID selected_mapping = 0;
            double mapping_cost = std::numeric_limits<double>::max();
            for (Medea::Individual::ID mapping_id = 0; mapping_id < mappings->second.size(); ++mapping_id)
            {
                // Thermal/power repair selects a low-energy mapping; SLO repair
                // selects a low-latency mapping on a compatible chiplet.
                const double cost = mappings->second[mapping_id].objectives[
                    prefer_coolest ? ENERGY_OBJ : LATENCY_OBJ];
                if (cost < mapping_cost)
                {
                    mapping_cost = cost;
                    selected_mapping = mapping_id;
                }
            }
            const double power = destination_id < ind.peak_chiplet_power.size() ? ind.peak_chiplet_power[destination_id] : 0.0;
            const double score = prefer_coolest
                ? power + 1e-12 * mapping_cost
                : mapping_cost * (1.0 + 0.01 * ind.accelerators[destination_id].num_assigned_layers);
            if (score < best_score)
            {
                best_score = score;
                best_destination = destination_id;
                best_mapping = selected_mapping;
            }
        }
        if (best_destination == source_id) return false;

        ind.genome[layer_id].mapping_id = best_mapping;
        ind.genome[layer_id].acc_instance_id = best_destination;
        ind.accelerators[source_id].num_assigned_layers--;
        ind.accelerators[best_destination].num_assigned_layers++;
        ind.accelerators[source_id].need_evaluation = true;
        ind.accelerators[best_destination].need_evaluation = true;
        ind.layer_evals[layer_id].need_evaluation = true;
        RemoveEmptySubAccelerators(ind);
        return true;
    }


    bool Moham::MoveLayerForBandwidthRelief(
        Individual& ind, Layer::ID layer_id) const
    {
        if (layer_id >= ind.genome.size() || ind.accelerators.size() < 2)
            return false;
        const ArchInstance::ID source_id = ind.genome[layer_id].acc_instance_id;
        const Workload::ID workload_id = (*graph_)[layer_id].workload_id;
        ArchInstance::ID best_destination = source_id;
        Medea::Individual::ID best_mapping = ind.genome[layer_id].mapping_id;
        double best_score = std::numeric_limits<double>::max();

        for (ArchInstance::ID destination_id = 0;
             destination_id < ind.accelerators.size(); ++destination_id)
        {
            if (destination_id == source_id) continue;
            const auto& destination = ind.accelerators[destination_id];
            const auto mappings =
                workload_mappings_[workload_id].find(destination.arch_template_id);
            if (mappings == workload_mappings_[workload_id].end() ||
                mappings->second.empty())
                continue;

            double route_hops = 0.0;
            if (destination.memory_interface_id < ind.memory_interfaces.size())
            {
                const auto memory_coordinates =
                    ind.memory_interfaces[destination.memory_interface_id].mesh_coordinates;
                route_hops =
                    std::abs(destination.mesh_coordinates.first - memory_coordinates.first) +
                    std::abs(destination.mesh_coordinates.second - memory_coordinates.second);
            }

            for (Medea::Individual::ID mapping_id = 0;
                 mapping_id < mappings->second.size(); ++mapping_id)
            {
                const auto& mapping = mappings->second[mapping_id];
                const auto& topology = mapping.engine.GetTopology();
                const auto arithmetic = mapping.arch.GetLevel(0);
                const double parallelism = static_cast<double>(std::max(
                    1, arithmetic.mesh_x * arithmetic.mesh_y));
                const double compute_cycles =
                    static_cast<double>(topology.ActualComputes()) / parallelism;
                const double bandwidth_proxy = compute_cycles > 0.0
                    ? static_cast<double>(topology.LastLevelAccesses()) /
                        compute_cycles
                    : std::numeric_limits<double>::max();
                const double score =
                    (1.0 + route_hops) * bandwidth_proxy +
                    0.05 * destination.num_assigned_layers;
                if (score < best_score)
                {
                    best_score = score;
                    best_destination = destination_id;
                    best_mapping = mapping_id;
                }
            }
        }
        if (best_destination == source_id) return false;

        ind.genome[layer_id].mapping_id = best_mapping;
        ind.genome[layer_id].acc_instance_id = best_destination;
        ind.accelerators[source_id].num_assigned_layers--;
        ind.accelerators[best_destination].num_assigned_layers++;
        ind.accelerators[source_id].need_evaluation = true;
        ind.accelerators[best_destination].need_evaluation = true;
        ind.layer_evals[layer_id].need_evaluation = true;
        RemoveEmptySubAccelerators(ind);
        return true;
    }


    bool Moham::ApplyBottleneckGuidedRepair(Individual& ind, std::mt19937_64& rng)
    {
        const bool random_policy = config_.moham.repair_policy == "random";
        if (ind.constraint_violation <= 1e-12 || ind.genome.empty() ||
            (!random_policy && ind.bottleneck_ledger.empty()) ||
            uni_distribution_(rng) >= config_.moham.bottleneck_repair_probability)
            return false;

        bool improved = false;
        for (unsigned attempt = 0;
             attempt < config_.moham.max_repair_attempts; ++attempt)
        {
            if (!random_policy && ind.bottleneck_ledger.empty()) break;
            BottleneckAttribution bottleneck = random_policy
                ? BottleneckAttribution{
                      static_cast<BottleneckKind>(
                          rng() % CONSTRAINT_METRIC_COUNT),
                      static_cast<Layer::ID>(rng() % ind.genome.size()),
                      0,
                      0.0}
                : ind.bottleneck_ledger.front();
            if (random_policy)
                bottleneck.accelerator_id =
                    ind.genome[bottleneck.layer_id].acc_instance_id;
            Individual candidate = ind;
            bool modified = false;
            switch (bottleneck.kind)
            {
                case BottleneckKind::SLO:
                    modified = MoveLayerToCompatibleAccelerator(candidate, bottleneck.layer_id, false);
                    break;
                case BottleneckKind::PACKAGE_POWER:
                case BottleneckKind::CHIPLET_POWER:
                case BottleneckKind::TEMPERATURE:
                    modified = MoveLayerToCompatibleAccelerator(candidate, bottleneck.layer_id, true);
                    break;
                case BottleneckKind::BANDWIDTH:
                    modified = MoveLayerForBandwidthRelief(
                        candidate, bottleneck.layer_id);
                    if (!modified && !candidate.toposort.empty())
                    {
                        const auto position = std::find(candidate.toposort.begin(), candidate.toposort.end(), bottleneck.layer_id);
                        if (position != candidate.toposort.end() && position != candidate.toposort.begin())
                        {
                            const auto index = std::distance(candidate.toposort.begin(), position);
                            std::size_t earliest = 0;
                            for (const auto& edge : (*graph_)[bottleneck.layer_id].incoming)
                            {
                                const auto predecessor = std::find(candidate.toposort.begin(), candidate.toposort.end(), edge.layer->id);
                                earliest = std::max(earliest, static_cast<std::size_t>(std::distance(candidate.toposort.begin(), predecessor) + 1));
                            }
                            if (earliest < static_cast<std::size_t>(index))
                            {
                                std::rotate(candidate.toposort.begin() + earliest, position, position + 1);
                                modified = true;
                            }
                        }
                    }
                    break;
                case BottleneckKind::AREA:
                    MergingMutation(candidate, rng);
                    modified = true;
                    break;
            }

            if (!modified || !CheckIndividualValidity(candidate, false)) break;
            unsigned remaining = repair_evaluation_budget_.load(std::memory_order_relaxed);
            while (remaining > 0 &&
                   !repair_evaluation_budget_.compare_exchange_weak(
                       remaining, remaining - 1, std::memory_order_relaxed))
            {
            }
            if (remaining == 0) break;
            repair_evaluation_count_.fetch_add(1, std::memory_order_relaxed);
            Evaluate(candidate, true);
            if (!ConstraintProfileDominates(candidate, ind)) break;
            ind = std::move(candidate);
            accepted_repair_evaluation_count_.fetch_add(
                1, std::memory_order_relaxed);
            improved = true;
        }
        return improved;
    }


    bool Moham::CheckIndividualValidity(Individual& ind, bool do_assert) const 
    {
        ind.valid = false;
        
        if (ind.accelerators.empty()) return false;

        std::unordered_set<ArchInstance::ID> arch_instance_ids_test_set;        

        for (Layer::ID l = 0; l < graph_->NumLayers(); l++) 
        {
            ArchInstance::ID instance_id = ind.genome[l].acc_instance_id;
            arch_instance_ids_test_set.insert(instance_id);

            Workload::ID workload_id = (*graph_)[l].workload_id;
            ArchTemplate::ID template_id = ind.accelerators[instance_id].arch_template_id;
            Medea::Individual::ID mapping_id = ind.genome[l].mapping_id;
            
            if (workload_mappings_.at(workload_id).find(template_id) == workload_mappings_.at(workload_id).end())
                return false;

            std::size_t possible_mappings = workload_mappings_.at(workload_id).at(template_id).size();

            if (mapping_id >= possible_mappings) 
                return false;

            // TMP for debug
            std::unordered_map<ArchInstance::ID, unsigned> num_assigned_layers;
            for (auto& gene : ind.genome)
                num_assigned_layers[gene.acc_instance_id] += 1;
            for (auto& acc : ind.accelerators)
                if (num_assigned_layers[acc.id] != acc.num_assigned_layers) {
                    if (do_assert) assert(num_assigned_layers[acc.id] == acc.num_assigned_layers); 
                    return false;
                }
        }

        if (arch_instance_ids_test_set.size() != ind.accelerators.size())
            return false;

        for (ArchInstance::ID a = 0; a < ind.accelerators.size(); a++)
            if (arch_instance_ids_test_set.find(a) == arch_instance_ids_test_set.end())
                return false;

        ind.valid = true;
        return true;
    }

    // TODO?? Layer* instead of Layer::ID for performance
    std::vector<Layer::ID> Moham::GenerateTopologicalSorting(const Individual& ind)
    {
        std::size_t num_layers = graph_->NumLayers();

        std::vector<Layer::ID> sorting;
        sorting.reserve(num_layers);

        std::unordered_set<Layer::ID> start_nodes;

        std::vector<int> in_degree(num_layers, 0);

        for (std::size_t l = 0; l < num_layers; l++) 
        {
            in_degree[l] = (*graph_)[l].incoming.size();
            if (in_degree[l] == 0) start_nodes.insert(l);
        }   
        
        while (!start_nodes.empty()) 
        {
            // Find start_node with highest priority
            auto mit = std::max_element(start_nodes.begin(), start_nodes.end(), [&ind](size_t a, size_t b){
                return ind.genome[a].priority > ind.genome[b].priority;
            });

            Layer::ID layer_id = *mit;
            sorting.push_back(layer_id);
            start_nodes.erase(mit);

            for (auto& e : (*graph_)[layer_id].outgoing)
                if (--in_degree[e.layer->id] == 0)
                    start_nodes.insert(e.layer->id);
        }

        assert(sorting.size() == num_layers);
        return sorting;
    }


    void Moham::CalculatePackageMesh(Individual& ind) 
    {
        std::size_t num_accelerators = ind.accelerators.size();

        ind.memory_interfaces.clear();

        unsigned mesh_x = std::ceil(std::sqrt(num_accelerators));
        unsigned mesh_y = std::ceil((double)num_accelerators / mesh_x);

        unsigned num_mis = config_.moham.max_memory_interfaces_amount;
        for (auto& am : config_.moham.memory_interfaces_amount)
            if (num_accelerators <= am.first)
                num_mis = am.second;

        if (num_mis > 2 * mesh_y) num_mis = 2 * mesh_y;

        for (ArchInstance::ID said = 0; said < ind.accelerators.size(); said++)
            ind.accelerators[said].mesh_coordinates = {said % mesh_x, said / mesh_x};

        for (MemoryInterface::ID mid = 0; mid < num_mis; mid++)
        {
            int x = (mid % 2) ? (mesh_x + 1) : (-1);
            int yt = mid / 2;
            int y = 0;
            if (config_.moham.memory_interfaces_position == "corner") {
                y = (yt % 2) ? (mesh_y - 1 - yt / 2) : (yt / 2);
            } else if (config_.moham.memory_interfaces_position == "middle") {
                int middle = (mesh_y - 1) / 2;
                y = (yt % 2) ? (middle + 1 + yt / 2) : (middle - yt / 2);
            } else {
                assert(false);
            }

            ind.memory_interfaces.push_back({mid, std::make_pair(x,y)});
        }

        for (ArchInstance::ID said = 0; said < ind.accelerators.size(); said++)
        {
            std::vector<MemoryInterface::ID> nearest_mis;
            int min_distance = std::numeric_limits<int>::max();
            for (MemoryInterface::ID mid = 0; mid < num_mis; mid++)
            {
                auto sac = ind.accelerators[said].mesh_coordinates;
                auto mic = ind.memory_interfaces[mid].mesh_coordinates;
                int distance = std::abs(sac.first - mic.first) + std::abs(sac.second - mic.second);
                if (distance == min_distance) {
                    nearest_mis.push_back(mid);
                } else if (distance < min_distance) {
                    nearest_mis = std::vector<MemoryInterface::ID>(1, mid);
                    min_distance = distance;
                }
            }
            ind.accelerators[said].memory_interface_id = *max_element(nearest_mis.begin(), nearest_mis.end());
        }

    }


    double Moham::EvaluateLayerCommunicationEnergy(
        const Individual& ind, Layer::ID layer_id) const
    {
        if (config_.moham.nip_hop_energy <= 0.0 ||
            layer_id >= ind.genome.size() || ind.memory_interfaces.empty())
            return 0.0;

        const ArchInstance::ID accelerator_id =
            ind.genome[layer_id].acc_instance_id;
        if (accelerator_id >= ind.accelerators.size())
            return 0.0;
        const auto& accelerator = ind.accelerators[accelerator_id];
        if (accelerator.memory_interface_id >= ind.memory_interfaces.size())
            return 0.0;

        const auto memory_coordinates =
            ind.memory_interfaces[accelerator.memory_interface_id].mesh_coordinates;
        const double distance =
            std::abs(accelerator.mesh_coordinates.first - memory_coordinates.first) +
            std::abs(accelerator.mesh_coordinates.second - memory_coordinates.second);
        const double transferred_words =
            ind.layer_evals[layer_id].cycles *
            ind.layer_evals[layer_id].required_bandwidth;
        return transferred_words * config_.moham.nip_hop_energy * distance;
    }


    double Moham::EvaluateNipEnergy(Individual& ind)
    {
        double total_hop_energy = 0.0;

        // Network-in-Package Energy Simulation
        if (config_.moham.nip_hop_energy > 0) {
            std::size_t num_layers = graph_->NumLayers();
            for (Layer::ID l = 0; l < num_layers; l++)
                total_hop_energy += EvaluateLayerCommunicationEnergy(ind, l);
        }

        return total_hop_energy;
    }


    double Moham::EvaluateMakespan(Individual& ind, const std::vector<Layer::ID>& topological_sorting)
    {
        const std::size_t num_layers = graph_->NumLayers();
        constexpr double epsilon = 1e-9;
        ind.compute_only_makespan = 0.0;
        ind.bandwidth_makespan_overhead = 0.0;
        ind.bandwidth_service_stretch_cycles = 0.0;
        ind.bandwidth_queue_delay_cycles = 0.0;
        ind.system_bandwidth_queue_delay_cycles = 0.0;
        ind.nip_bandwidth_queue_delay_cycles = 0.0;
        ind.bandwidth_stretched_layers = 0;
        ind.bandwidth_delayed_layers = 0;

        // Counterfactual schedule for the same design, accelerator assignment,
        // and topological order with communication capacity removed. This is a
        // diagnostic baseline only and never feeds back into optimization.
        std::vector<double> compute_only_finishing_times(num_layers, 0.0);
        std::unordered_map<ArchInstance::ID, double> compute_only_instance_finishing_times;
        for (Layer::ID l : topological_sorting)
        {
            const Layer& layer = (*graph_)[l];
            const Gene& gene = ind.genome[l];
            double earliest_start = 0.0;
            for (const auto& dep : layer.incoming)
                earliest_start = std::max(
                    compute_only_finishing_times[dep.layer->id], earliest_start);
            earliest_start = std::max(
                compute_only_instance_finishing_times[gene.acc_instance_id],
                earliest_start);
            const double duration = std::max(
                1.0, std::ceil(ind.layer_evals[l].cycles - epsilon));
            const double end_time = earliest_start + duration;
            compute_only_finishing_times[l] = end_time;
            compute_only_instance_finishing_times[gene.acc_instance_id] = end_time;
            ind.compute_only_makespan = std::max(
                ind.compute_only_makespan, end_time);
        }

        std::vector<double> layer_finishing_times(num_layers, 0.0);
        std::unordered_map<ArchInstance::ID, double> instances_finishing_times;
        std::vector<double> effective_bandwidth(num_layers, 0.0);
        std::vector<std::vector<MeshLink>> routes(num_layers);
        std::vector<Layer::ID> scheduled_layers;
        scheduled_layers.reserve(num_layers);
        ind.nip_bottlenecks.clear();
        ind.memory_bottlenecks.clear();

        for (Layer::ID l : topological_sorting)
        {
            const Layer& layer = (*graph_)[l];
            const Gene& gene = ind.genome[l];
            const auto& accelerator = ind.accelerators[gene.acc_instance_id];
            if (accelerator.memory_interface_id < ind.memory_interfaces.size())
                routes[l] = BuildXYRoute(
                    ind.memory_interfaces[accelerator.memory_interface_id].mesh_coordinates,
                    accelerator.mesh_coordinates);

            const double transferred_words =
                ind.layer_evals[l].cycles * ind.layer_evals[l].required_bandwidth;
            double duration = ind.layer_evals[l].cycles;
            if (config_.moham.system_bandwidth > 0.0)
                duration = std::max(
                    duration, transferred_words / config_.moham.system_bandwidth);
            if (config_.moham.nip_link_bandwidth > 0.0 && !routes[l].empty())
                duration = std::max(
                    duration, transferred_words / config_.moham.nip_link_bandwidth);
            duration = std::max(1.0, std::ceil(duration - epsilon));
            const double compute_only_duration = std::max(
                1.0, std::ceil(ind.layer_evals[l].cycles - epsilon));
            const double service_stretch = std::max(
                0.0, duration - compute_only_duration);
            ind.bandwidth_service_stretch_cycles += service_stretch;
            if (service_stretch > epsilon)
                ++ind.bandwidth_stretched_layers;
            effective_bandwidth[l] =
                duration > 0.0 ? transferred_words / duration : 0.0;

            double earliest_start = 0.0;
            for (auto& dep : layer.incoming)
                earliest_start = std::max(
                    layer_finishing_times[dep.layer->id], earliest_start);
            earliest_start = std::max(
                instances_finishing_times[gene.acc_instance_id], earliest_start);

            double start_time = earliest_start;
            bool delayed_by_system = false;
            bool delayed_by_link = false;
            for (std::size_t attempt = 0; attempt <= scheduled_layers.size(); ++attempt)
            {
                const double end_time = start_time + duration;
                std::vector<double> boundaries{start_time, end_time};
                for (Layer::ID scheduled : scheduled_layers)
                {
                    const auto& evaluation = ind.layer_evals[scheduled];
                    if (start_time < evaluation.start_time && evaluation.start_time < end_time)
                        boundaries.push_back(evaluation.start_time);
                    if (start_time < evaluation.end_time && evaluation.end_time < end_time)
                        boundaries.push_back(evaluation.end_time);
                }
                std::sort(boundaries.begin(), boundaries.end());
                boundaries.erase(
                    std::unique(boundaries.begin(), boundaries.end()),
                    boundaries.end());

                bool conflict = false;
                double next_start = std::numeric_limits<double>::max();
                for (std::size_t interval = 0;
                     interval + 1 < boundaries.size() && !conflict; ++interval)
                {
                    const double interval_start = boundaries[interval];
                    const double interval_end = boundaries[interval + 1];
                    if (interval_end <= interval_start) continue;

                    double system_demand = effective_bandwidth[l];
                    std::map<MeshLink, double> link_demand;
                    for (const auto& link : routes[l])
                        link_demand[link] += effective_bandwidth[l];
                    std::vector<Layer::ID> active;
                    for (Layer::ID scheduled : scheduled_layers)
                    {
                        const auto& evaluation = ind.layer_evals[scheduled];
                        if (evaluation.start_time <= interval_start &&
                            interval_end <= evaluation.end_time)
                        {
                            active.push_back(scheduled);
                            system_demand += effective_bandwidth[scheduled];
                            for (const auto& link : routes[scheduled])
                                link_demand[link] += effective_bandwidth[scheduled];
                        }
                    }

                    const bool system_conflict =
                        config_.moham.system_bandwidth > 0.0 &&
                        system_demand > config_.moham.system_bandwidth + epsilon;
                    bool link_conflict = false;
                    if (config_.moham.nip_link_bandwidth > 0.0)
                        for (const auto& demand : link_demand)
                            if (demand.second >
                                config_.moham.nip_link_bandwidth + epsilon)
                            {
                                link_conflict = true;
                                break;
                            }
                    if (system_conflict || link_conflict)
                    {
                        conflict = true;
                        delayed_by_system = delayed_by_system || system_conflict;
                        delayed_by_link = delayed_by_link || link_conflict;
                        for (Layer::ID active_layer : active)
                            next_start = std::min(
                                next_start, ind.layer_evals[active_layer].end_time);
                    }
                }

                if (!conflict) break;
                if (!std::isfinite(next_start) || next_start <= start_time + epsilon)
                    throw std::runtime_error(
                        "resource scheduler could not advance from a bandwidth conflict");
                start_time = next_start;
            }

            const double end_time = start_time + duration;
            const double queue_delay = std::max(0.0, start_time - earliest_start);
            if ((delayed_by_system || delayed_by_link) && queue_delay > epsilon)
            {
                ind.bandwidth_queue_delay_cycles += queue_delay;
                ++ind.bandwidth_delayed_layers;
            }
            if (delayed_by_system)
                ind.system_bandwidth_queue_delay_cycles += queue_delay;
            if (delayed_by_link)
                ind.nip_bandwidth_queue_delay_cycles += queue_delay;
            if (delayed_by_link)
                ind.nip_bottlenecks.push_back(
                    {earliest_start, start_time, gene.acc_instance_id, l});
            if (delayed_by_system)
                ind.memory_bottlenecks.push_back(
                    {earliest_start, start_time, gene.acc_instance_id, l});
            layer_finishing_times[layer.id] = end_time;
            instances_finishing_times[gene.acc_instance_id] = end_time;
            ind.layer_evals[l].start_time = start_time;
            ind.layer_evals[l].end_time = end_time;
            scheduled_layers.push_back(l);
        }

        double makespan = 0.0;
        for (Layer::ID l = 0; l < num_layers; l++)
            if (ind.layer_evals[l].end_time > makespan)
                makespan = ind.layer_evals[l].end_time;

        ind.bandwidth_makespan_overhead = std::max(
            0.0, makespan - ind.compute_only_makespan);

        return makespan;
    }


    void Moham::NegotiateArchitectures(Individual& ind)
    {
        if (config_.moham.negotiate_arch) 
        {
            std::unordered_map<ArchInstance::ID, MinimalArchSpecs> new_negotiated_archs;

            for (size_t l = 0; l < graph_->NumLayers(); l++)
            {
                ArchInstance::ID acc_instance_id = ind.genome[l].acc_instance_id;
                ArchTemplate::ID template_id = ind.accelerators[acc_instance_id].arch_template_id;
                
                Workload::ID workload_id = (*graph_)[l].workload_id;
                Medea::Individual::ID mapping_id = ind.genome[l].mapping_id;

                auto ait = new_negotiated_archs.find(acc_instance_id);

                if (ait != new_negotiated_archs.end()) {
                    ait->second &= workload_mappings_[workload_id][template_id][mapping_id].arch;
                } else { 
                    new_negotiated_archs[acc_instance_id] = workload_mappings_[workload_id][template_id][mapping_id].arch;
                }
            }

            ind.accelerators.resize(new_negotiated_archs.size());
            for (ArchInstance::ID instance_id = 0; instance_id < new_negotiated_archs.size(); instance_id++)
            {
                if (ind.accelerators[instance_id].arch != new_negotiated_archs.at(instance_id))
                {
                    ind.accelerators[instance_id].arch = new_negotiated_archs.at(instance_id);
                    ind.accelerators[instance_id].need_evaluation = true;
                
                    ArchTemplate& arch_template = (*arch_templates_)[ind.accelerators[instance_id].arch_template_id];

                    auto new_specs = timeloop::TopologySpecs(arch_template.arch_specs.topology); 

                    auto minimal_arithmetic = ind.accelerators[instance_id].arch.GetLevel(0);
                    auto arithmetic = new_specs.GetArithmeticLevel();
                    arithmetic->meshX = minimal_arithmetic.mesh_x;
                    arithmetic->meshY = minimal_arithmetic.mesh_y;
                    arithmetic->instances = minimal_arithmetic.mesh_x * minimal_arithmetic.mesh_y;

                    std::unordered_map<std::string, uint64_t> updates;

                    for (unsigned i = 1; i < arch_template.arch_specs.topology.NumLevels(); i++)
                    {
                        auto buffer = new_specs.GetStorageLevel(i - 1);
                        if (!buffer->size.IsSpecified())
                            continue;

                        auto minimal_buffer = ind.accelerators[instance_id].arch.GetLevel(i);
                        buffer->meshX = minimal_buffer.mesh_x;
                        buffer->meshY = minimal_buffer.mesh_y;
                        buffer->instances = minimal_buffer.mesh_x * minimal_buffer.mesh_y;
                        buffer->size = minimal_buffer.size;
                        buffer->effective_size = static_cast<uint64_t>(std::floor(minimal_buffer.size / buffer->multiple_buffering.Get()));

                        updates[buffer->name.Get()] = buffer->size.Get() / buffer->block_size.Get();
                    }

                    std::string out_prefix = "moham." + std::to_string(omp_get_thread_num()) + "_tmp";
                    Accelergy::RT rt = arch_template.accelergy.GetReferenceTables(updates, out_prefix);

                    ind.accelerators[instance_id].arch_specs = (timeloop::EngineSpecs){.topology = new_specs};
                    ind.accelerators[instance_id].arch_specs.topology.ParseAccelergyART(rt.area);
                    ind.accelerators[instance_id].arch_specs.topology.ParseAccelergyERT(rt.energy);
                }
            }
        } 
        else 
        { 
            for (ArchInstance::ID instance_id = 0; instance_id < ind.accelerators.size(); instance_id++)
            {
                ArchTemplate& arch_template = (*arch_templates_)[ind.accelerators[instance_id].arch_template_id];
                auto new_specs = timeloop::TopologySpecs(arch_template.arch_specs.topology); 

                std::unordered_map<std::string, uint64_t> updates;

                for (unsigned i = 1; i < arch_template.arch_specs.topology.NumLevels(); i++)
                {
                    auto buffer = new_specs.GetStorageLevel(i - 1);
                    if (!buffer->size.IsSpecified())
                        continue;

                    updates[buffer->name.Get()] = buffer->size.Get() / buffer->block_size.Get();
                }

                std::string out_prefix = "moham." + std::to_string(omp_get_thread_num()) + "_tmp";
                Accelergy::RT rt = arch_template.accelergy.GetReferenceTables(updates, out_prefix);

                ind.accelerators[instance_id].arch_specs = (timeloop::EngineSpecs){.topology = new_specs};
                ind.accelerators[instance_id].arch_specs.topology.ParseAccelergyART(rt.area);
                ind.accelerators[instance_id].arch_specs.topology.ParseAccelergyERT(rt.energy);
            }
        }
    }


    std::size_t Moham::Tournament(const Population& pop)
    {
        auto& rng =  rng_[omp_get_thread_num()];
        std::size_t b1 = rng() % pop.size();
        std::size_t b2 = rng() % pop.size();

        if (pop[b1].rank < pop[b2].rank)
        {
            return b1;
        }
        else if (pop[b1].rank == pop[b2].rank)
        {
            if (pop[b1].crowding_distance > pop[b2].crowding_distance)
                return b1;
            else
                return b2;
        }
        else
        {
            return b2;
        }
    }


    void Moham::Crossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b)
    {
        offspring_a = parent_a;
        offspring_b = parent_b;

        auto& rng =  rng_[omp_get_thread_num()];

        if (uni_distribution_(rng) < config_.moham.priority_crossover_prob) 
            PriorityCrossover(parent_a, parent_b, offspring_a, offspring_b, rng);

        if (uni_distribution_(rng) < config_.moham.subacc_crossover_prob) 
            SubAcceleratorCrossover(parent_a, parent_b, offspring_a, offspring_b, rng);

        if (uni_distribution_(rng) < config_.moham.mapping_crossover_prob) 
            MappingCrossover(parent_a, parent_b, offspring_a, offspring_b, rng);

        // General crossover
        // std::swap_ranges(offspring_a.genome.begin(), offspring_a.genome.begin() + crossover_point, offspring_b.genome.begin());
    }


    void Moham::PriorityCrossover(const Individual& pa, const Individual& pb, Individual& osa, Individual& osb, std::mt19937_64& rng)
    {
        std::size_t num_layers = graph_->NumLayers();
        std::size_t crossover_point = rng() % num_layers;

        if (!config_.moham.xu_priority) {
            for (Layer::ID l = 0; l < crossover_point; l++) 
                std::swap(osb.genome[l].priority, osa.genome[l].priority);
            return;
        }

        Layer::ID pta = crossover_point;
        Layer::ID ptb = crossover_point;

        for (Layer::ID l = 0; l < num_layers; l++)
        {
            if (std::find(pa.toposort.begin(), pa.toposort.begin() + crossover_point, pb.toposort[l]) == (pa.toposort.begin() + crossover_point))
                osa.toposort[pta++] = pb.toposort[l];
            
            if (std::find(pb.toposort.begin(), pb.toposort.begin() + crossover_point, pa.toposort[l]) == (pb.toposort.begin() + crossover_point))
                osb.toposort[ptb++] = pa.toposort[l];
        }

        assert(pta == num_layers && ptb == num_layers);
    }


    void Moham::SubAcceleratorCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng)
    {
        std::size_t num_layers = graph_->NumLayers();
        ArchInstance::ID chosen_acc_id = rng() % parent_a.accelerators.size();

        // Moving accelerator scheduling from A to B
        if (chosen_acc_id < parent_b.accelerators.size())
        {
            // Accelerator already exists in B
            std::swap(offspring_a.accelerators[chosen_acc_id], offspring_b.accelerators[chosen_acc_id]);
            
            // Moving accelerator scheduling from B to A
            for (Layer::ID l = 0; l < num_layers; l++) 
            {
                if (parent_b.genome[l].acc_instance_id == chosen_acc_id)
                {
                    offspring_a.accelerators[offspring_a.genome[l].acc_instance_id].need_evaluation = true;

                    offspring_a.genome[l].mapping_id = parent_b.genome[l].mapping_id;
                    offspring_a.genome[l].acc_instance_id = parent_b.genome[l].acc_instance_id;
                    offspring_a.layer_evals[l].need_evaluation = true;

                }
            }
            // Moving A to B
            for (Layer::ID l = 0; l < num_layers; l++) 
            {
                if (parent_a.genome[l].acc_instance_id == chosen_acc_id)
                {
                    offspring_b.accelerators[offspring_b.genome[l].acc_instance_id].need_evaluation = true;

                    offspring_b.genome[l].mapping_id = parent_a.genome[l].mapping_id;
                    offspring_b.genome[l].acc_instance_id = parent_a.genome[l].acc_instance_id;
                    offspring_b.layer_evals[l].need_evaluation = true;
                }
            } 

            // Count assigned layer to each accelerator
            for (auto& acc : offspring_a.accelerators) acc.num_assigned_layers = 0;
            for (auto& acc : offspring_b.accelerators) acc.num_assigned_layers = 0;
            for (Layer::ID l = 0; l < num_layers; l++) {
                offspring_a.accelerators[offspring_a.genome[l].acc_instance_id].num_assigned_layers++;
                offspring_b.accelerators[offspring_b.genome[l].acc_instance_id].num_assigned_layers++;
            }

            offspring_a.accelerators[chosen_acc_id].need_evaluation = true;
            offspring_b.accelerators[chosen_acc_id].need_evaluation = true;

            RemoveEmptySubAccelerators(offspring_a);
            RemoveEmptySubAccelerators(offspring_b);
        } 
        else 
        {
            // Accelerator doesnt exist in B, we add it
            ArchInstance::ID new_instance_id = offspring_b.accelerators.size();
            offspring_b.accelerators.push_back(parent_a.accelerators[chosen_acc_id]);
            offspring_b.accelerators.back().id = new_instance_id;

            for (Layer::ID l = 0; l < num_layers; l++) 
            {
                if (parent_a.genome[l].acc_instance_id == chosen_acc_id)
                {
                    offspring_b.accelerators[offspring_b.genome[l].acc_instance_id].need_evaluation = true;

                    offspring_b.genome[l].mapping_id = parent_a.genome[l].mapping_id;
                    offspring_b.genome[l].acc_instance_id = new_instance_id;
                    offspring_b.layer_evals[l].need_evaluation = true;
                }
            }

            // Count assigned layer to each accelerator
            for (auto& acc : offspring_b.accelerators) acc.num_assigned_layers = 0;
            for (Layer::ID l = 0; l < num_layers; l++) {
                offspring_b.accelerators[offspring_b.genome[l].acc_instance_id].num_assigned_layers++;
            }

            offspring_b.accelerators[new_instance_id].need_evaluation = true;
            RemoveEmptySubAccelerators(offspring_b);
        }


        // TODO Repair when templates are different
    }


    void Moham::MappingCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng)
    {
        std::size_t crossover_point = rng() % graph_->NumLayers();

        for (Layer::ID l = 0; l < crossover_point; l++) 
        {
            ArchTemplate::ID template_a = parent_a.accelerators[parent_a.genome[l].acc_instance_id].arch_template_id;
            ArchTemplate::ID template_b = parent_b.accelerators[parent_b.genome[l].acc_instance_id].arch_template_id;
            Workload::ID workload_id = (*graph_)[l].workload_id;

            if (template_a == template_b)
            {
                std::swap(offspring_a.genome[l].mapping_id, offspring_b.genome[l].mapping_id);
            } 
            else 
            {
                offspring_a.genome[l].mapping_id = ConvertMappingToTemplate(workload_id, parent_b.genome[l].mapping_id, template_b, template_a);
                offspring_b.genome[l].mapping_id = ConvertMappingToTemplate(workload_id, parent_a.genome[l].mapping_id, template_a, template_b);
            }

            offspring_a.layer_evals[l].need_evaluation = true;
            offspring_b.layer_evals[l].need_evaluation = true;        
        }
    }
    

    void Moham::Mutation(Individual& ind)
    {
        auto& rng =  rng_[omp_get_thread_num()];
        std::size_t num_layers = graph_->NumLayers();

        if (uni_distribution_(rng) < config_.moham.splitting_mutation_prob)
            SplittingMutation(ind, rng);

        if (uni_distribution_(rng) < config_.moham.merging_mutation_prob)
            MergingMutation(ind, rng);

        if (uni_distribution_(rng) < config_.moham.template_mutation_prob)
            TemplateMutation(ind, rng);

        if (uni_distribution_(rng) < config_.moham.position_mutation_prob)
            PositionMutation(ind, rng);

        for (Layer::ID l = 0; l < num_layers; l++)
        {
            if (uni_distribution_(rng) < config_.moham.priority_mutation_prob)
                PriorityMutation(ind, l, rng);

            if (uni_distribution_(rng) < config_.moham.mapping_mutation_prob)
                MappingMutation(ind, l, rng);

            if (uni_distribution_(rng) < config_.moham.assignment_mutation_prob)
                AssignmentMutation(ind, l, rng);
        }
    }
    

    void Moham::SplittingMutation(Individual& ind, std::mt19937_64& rng)
    {
        if (ind.accelerators.size() >= config_.moham.max_subaccelerators)
            return;
        
        ArchInstance::ID acc_to_split = rng() % ind.accelerators.size();
        ArchInstance::ID new_acc_id = ind.accelerators.size();

        ind.accelerators.push_back(ind.accelerators[acc_to_split]);
        ind.accelerators[new_acc_id].id = new_acc_id;
        
        for (Layer::ID l = 0; l < graph_->NumLayers(); l++)
            if (ind.genome[l].acc_instance_id == acc_to_split)
            {
                if (uni_distribution_(rng) < 0.5) {
                    ind.genome[l].acc_instance_id = new_acc_id;
                    ind.accelerators[acc_to_split].num_assigned_layers--;
                } else {
                    ind.accelerators[new_acc_id].num_assigned_layers--;
                }
            }
        
        RemoveEmptySubAccelerators(ind); 
    }


    void Moham::MergingMutation(Individual& ind, std::mt19937_64& rng)
    {
        if (ind.accelerators.size() <= 1) return;
        ArchInstance::ID id_a = rng() % ind.accelerators.size();
        ArchInstance::ID id_b = rng() % ind.accelerators.size();
        if (id_a == id_b) return;
        
        ArchInstance::ID destination_id = std::min(id_a, id_b);
        ArchInstance::ID source_id = std::max(id_a, id_b);

        ArchTemplate::ID destination_template = ind.accelerators[destination_id].arch_template_id;
        ArchTemplate::ID source_template = ind.accelerators[source_id].arch_template_id;

        ind.accelerators[destination_id].num_assigned_layers += ind.accelerators[source_id].num_assigned_layers;
        ind.accelerators[source_id].num_assigned_layers = 0;
        
        for (Layer::ID l = 0; l < ind.genome.size(); l++)
            if (ind.genome[l].acc_instance_id == source_id) 
            {
                ind.genome[l].acc_instance_id = destination_id;

                if (destination_template != source_template && 
                    workload_mappings_[(*graph_)[l].workload_id][destination_template].size() > 0 )
                    ind.genome[l].mapping_id = ConvertMappingToTemplate((*graph_)[l].workload_id, ind.genome[l].mapping_id, source_template, destination_template);
            } 

        RemoveEmptySubAccelerators(ind);           

    }


    void Moham::TemplateMutation(Individual& ind, std::mt19937_64& rng)
    {
        ArchInstance::ID acc_id = rng() % ind.accelerators.size();
        ArchTemplate::ID start_template = ind.accelerators[acc_id].arch_template_id;

        ArchTemplate::ID destination_template = rng() % arch_templates_->size();
        if (start_template == destination_template) return;

        ind.accelerators[acc_id].arch_template_id = destination_template;
        for (Layer::ID l = 0; l < ind.genome.size(); l++)
            if (ind.genome[l].acc_instance_id == acc_id) 
                if (destination_template != start_template && 
                    workload_mappings_[(*graph_)[l].workload_id][destination_template].size() > 0 ) 
                    ind.genome[l].mapping_id = ConvertMappingToTemplate((*graph_)[l].workload_id, ind.genome[l].mapping_id, start_template, destination_template);
        
        ind.accelerators[acc_id].need_evaluation = true;
    }


    void Moham::PositionMutation(Individual& ind, std::mt19937_64& rng)
    {
        if (ind.accelerators.size() <= 1) return;

        ArchInstance::ID acc_id_a = rng() % ind.accelerators.size();
        ArchInstance::ID acc_id_b = rng() % ind.accelerators.size();

        if (acc_id_a == acc_id_b) return;

        for (Layer::ID l = 0; l < graph_->NumLayers(); l++)
        {
            if (ind.genome[l].acc_instance_id == acc_id_a)
            {
                ind.genome[l].acc_instance_id = acc_id_b;
            }
            else if (ind.genome[l].acc_instance_id == acc_id_b)
            {
                ind.genome[l].acc_instance_id = acc_id_a;
            }
        }

        std::swap(ind.accelerators[acc_id_a], ind.accelerators[acc_id_b]);
        std::swap(ind.accelerators[acc_id_a].id, ind.accelerators[acc_id_b].id);
    }


    void Moham::PriorityMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng)
    {
        if (!config_.moham.xu_priority) {
            ind.genome[l].priority = uni_distribution_(rng);
            return;
        }

        Layer::ID succ = std::numeric_limits<std::size_t>::max();
        for (auto& out_edge : (*graph_)[ind.toposort[l]].outgoing)
        {
            auto itr = std::find(ind.toposort.begin(), ind.toposort.end(), out_edge.layer->id);
            Layer::ID new_succ = std::distance(ind.toposort.begin(), itr);
            succ = std::min(new_succ, succ);
        }
        
        int span = (int)succ - l - 1;
        if (succ >= ind.toposort.size() || span <= 0) return;

        Layer::ID k = l + 1 + (rng() % span);
        for (auto& in_edge : (*graph_)[ind.toposort[k]].incoming)
        {
            auto itr = std::find(ind.toposort.begin(), ind.toposort.end(), in_edge.layer->id);
            Layer::ID pred = std::distance(ind.toposort.begin(), itr);
            if (pred > l) return;
        }

        std::swap(ind.toposort[l], ind.toposort[k]);
    }


    void Moham::MappingMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng)
    {
        Workload::ID workload_id = (*graph_)[l].workload_id;
        ArchTemplate::ID template_id = ind.accelerators.at(ind.genome[l].acc_instance_id).arch_template_id;

        auto& possible_mapings = workload_mappings_[workload_id][template_id];
        std::size_t num_possible_mappings = possible_mapings.size();
        if (num_possible_mappings == 0) return; // invalid mapping
        
        ind.genome[l].mapping_id = rng() % num_possible_mappings;
    }


    void Moham::AssignmentMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng)
    {
        Medea::Individual::ID mapping_id = ind.genome[l].mapping_id;
        Workload::ID workload_id = (*graph_)[l].workload_id;
        ArchInstance::ID starting_instance_id = ind.genome[l].acc_instance_id;
        ArchTemplate::ID starting_template = ind.accelerators[starting_instance_id].arch_template_id;

        if ( mapping_id >= workload_mappings_[workload_id][starting_template].size() ) return; // invalid mapping

        std::size_t max_instance_id = std::min(ind.accelerators.size()+1, (std::size_t)config_.moham.max_subaccelerators);
        ArchInstance::ID destination_instance_id = rng() % max_instance_id;

        if (destination_instance_id == ind.accelerators.size()) 
        {
            ind.accelerators.push_back(ind.accelerators[starting_instance_id]);
            ind.accelerators.back().id = destination_instance_id;
            ind.accelerators.back().num_assigned_layers = 1;

            ind.genome[l].acc_instance_id = destination_instance_id;

            ind.accelerators[starting_instance_id].num_assigned_layers--;
        } 
        else 
        { 
            ArchTemplate::ID destination_template = ind.accelerators[destination_instance_id].arch_template_id;
            if (workload_mappings_[workload_id][destination_template].size() > 0) 
            {
                ind.genome[l].acc_instance_id = destination_instance_id;

                if (starting_template != destination_template)
                    ind.genome[l].mapping_id = ConvertMappingToTemplate(workload_id, mapping_id, starting_template, destination_template);
            
                ind.accelerators[starting_instance_id].num_assigned_layers--;
                ind.accelerators[destination_instance_id].num_assigned_layers++;
            }
        }

        RemoveEmptySubAccelerators(ind);
    }


    void Moham::RemoveEmptySubAccelerators(Individual& ind) const 
    {
        for(auto ita = ind.accelerators.begin(); ita != ind.accelerators.end(); )
        {
            if (ita->num_assigned_layers == 0)
            {
                ita = ind.accelerators.erase(ita); 
                for (auto itar = ita; itar != ind.accelerators.end(); itar++)
                    itar->id--;
                for (Layer::ID l = 0; l < ind.genome.size(); l++)
                    if ( ind.genome[l].acc_instance_id > (ArchInstance::ID)(ita - ind.accelerators.begin()) )
                        ind.genome[l].acc_instance_id--;
            }
            else
            {
                ++ita;
            }
        }
    }


    Medea::Individual::ID Moham::ConvertMappingToTemplate(Workload::ID workload_id, Medea::Individual::ID mapping_id, ArchTemplate::ID start_template, ArchTemplate::ID destination_template) const 
    {   
        if (mapping_id >= workload_mappings_[workload_id].at(start_template).size() ) return 0;

        auto cur_normalized_objectives = workload_mappings_normalized_objectives_[workload_id].at(start_template).at(mapping_id);
        Medea::Individual::ID nearest_point_id = 0;
        double minimum_distance = std::numeric_limits<double>::max();

        for (Medea::Individual::ID i = 0; i < workload_mappings_normalized_objectives_[workload_id].at(destination_template).size(); i++)
        {
            double distance = 0.0;
            for (unsigned o = 0; o < workload_mappings_normalized_objectives_[workload_id].at(destination_template)[i].size(); o++) {
                double obj = workload_mappings_normalized_objectives_[workload_id].at(destination_template)[i][o];
                distance += (obj-cur_normalized_objectives[o]) * (obj-cur_normalized_objectives[o]);
            }
            if (distance < minimum_distance)
            {
                minimum_distance = distance;
                nearest_point_id = i;
            }
        }

        return nearest_point_id;
    }


    void Moham::OutputParetoScheduling(std::ofstream &out, const Population& pop) 
    {
        out << "individual,layer,network,accelerator,template,mesh_x,mesh_y,accelerator_area,start_time,end_time,energy,communication_energy,cycles,layer_area,toposort" << std::endl;
        for (std::size_t i = 0, j = 0; i < pop.size(); i++)
        {
            if (pop[i].rank == 0) 
            {
                std::string toposort = std::to_string(pop[i].toposort[0]);
                for (Layer::ID l = 1; l < pop[i].toposort.size(); l++)
                    toposort += "-" + std::to_string(pop[i].toposort[l]);

                for (Layer::ID l = 0; l < graph_->NumLayers(); l++) 
                {           
                    out << j;
                    out << "," << l;
                    out << "," << (*graph_)[l].network;
                    out << "," << pop[i].genome[l].acc_instance_id;
                    const auto& accelerator =
                        pop[i].accelerators[pop[i].genome[l].acc_instance_id];
                    out << "," << (*arch_templates_)[accelerator.arch_template_id].name;
                    out << "," << accelerator.mesh_coordinates.first;
                    out << "," << accelerator.mesh_coordinates.second;
                    out << "," << accelerator.area;
                    out << "," << pop[i].layer_evals[l].start_time;
                    out << "," << pop[i].layer_evals[l].end_time;
                    out << "," << pop[i].layer_evals[l].energy;
                    out << "," << EvaluateLayerCommunicationEnergy(pop[i], l);
                    out << "," << pop[i].layer_evals[l].cycles;
                    out << "," << pop[i].layer_evals[l].area;
                    out << "," << toposort;
                    out << std::endl;
                }
                j++;
            }
        }
    }


    void Moham::OutputParetoNipBottlenecks(std::ofstream &out, const Population& pop) {
        out << "individual,resource,accelerator,layer,start_time,end_time" << std::endl;
        for (std::size_t i = 0, j = 0; i < pop.size(); i++)
        {
            if (pop[i].rank == 0) 
            {
                for (auto& bottle : pop[i].nip_bottlenecks) 
                {           
                    out << j;
                    out << ",nip_link";
                    out << "," << bottle.acc;
                    out << "," << bottle.layer_id;
                    out << "," << bottle.start_time;
                    out << "," << bottle.end_time;
                    out << std::endl;
                }
                for (auto& bottle : pop[i].memory_bottlenecks)
                {
                    out << j;
                    out << ",dram";
                    out << "," << bottle.acc;
                    out << "," << bottle.layer_id;
                    out << "," << bottle.start_time;
                    out << "," << bottle.end_time;
                    out << std::endl;
                }
                j++;
            }
        }
    }


    void Moham::OutputParetoMetrics(std::ofstream &out, const Population& pop)
    {
        out << std::setprecision(17);
        out << "individual,design_fingerprint,energy_pj,makespan_cycles,area_um2,peak_temperature_c,slo_risk,peak_package_power_w,peak_chiplet_power_w,peak_system_bandwidth,peak_nip_bandwidth,network_completion_cycles,compute_only_makespan_cycles,bandwidth_makespan_overhead_cycles,bandwidth_makespan_overhead_ratio,bandwidth_service_stretch_cycles,bandwidth_queue_delay_cycles,system_bandwidth_queue_delay_cycles,nip_bandwidth_queue_delay_cycles,bandwidth_stretched_layers,bandwidth_delayed_layers,constraint_profile,constraint_violation,feasible" << std::endl;
        for (std::size_t i = 0, output_id = 0; i < pop.size(); ++i)
        {
            if (pop[i].rank != 0) continue;
            out << output_id++;
            out << "," << pop[i].design_fingerprint;
            out << "," << pop[i].objectives[0];
            out << "," << pop[i].objectives[1];
            out << "," << pop[i].objectives[2];
            out << "," << pop[i].objectives[3];
            out << "," << pop[i].objectives[4];
            out << "," << pop[i].peak_package_power;
            out << ",\"";
            for (std::size_t acc_id = 0; acc_id < pop[i].peak_chiplet_power.size(); ++acc_id)
            {
                if (acc_id) out << ";";
                out << pop[i].peak_chiplet_power[acc_id];
            }
            out << "\"";
            out << "," << pop[i].peak_system_bandwidth;
            out << "," << pop[i].peak_nip_bandwidth;
            out << ",\"";
            for (std::size_t network_id = 0; network_id < pop[i].network_completion_times.size(); ++network_id)
            {
                if (network_id) out << ";";
                out << pop[i].network_completion_times[network_id];
            }
            out << "\"";
            out << "," << pop[i].compute_only_makespan;
            out << "," << pop[i].bandwidth_makespan_overhead;
            out << "," << pop[i].bandwidth_makespan_overhead /
                std::max(pop[i].objectives[1], 1.0);
            out << "," << pop[i].bandwidth_service_stretch_cycles;
            out << "," << pop[i].bandwidth_queue_delay_cycles;
            out << "," << pop[i].system_bandwidth_queue_delay_cycles;
            out << "," << pop[i].nip_bandwidth_queue_delay_cycles;
            out << "," << pop[i].bandwidth_stretched_layers;
            out << "," << pop[i].bandwidth_delayed_layers;
            out << ",\"";
            for (std::size_t metric = 0; metric < CONSTRAINT_METRIC_COUNT; ++metric)
            {
                if (metric) out << ";";
                out << pop[i].constraint_profile[metric];
            }
            out << "\"";
            out << "," << pop[i].constraint_violation;
            out << "," << (pop[i].constraint_violation <= 1e-12 ? "true" : "false") << std::endl;
        }
    }


    void Moham::OutputFinalPopulationMetrics(std::ofstream &out, const Population& pop)
    {
        out << std::setprecision(17);
        out << "individual,rank,design_fingerprint,energy_pj,makespan_cycles,area_um2,peak_temperature_c,slo_risk,compute_only_makespan_cycles,bandwidth_makespan_overhead_cycles,bandwidth_makespan_overhead_ratio,bandwidth_service_stretch_cycles,bandwidth_queue_delay_cycles,system_bandwidth_queue_delay_cycles,nip_bandwidth_queue_delay_cycles,bandwidth_stretched_layers,bandwidth_delayed_layers,constraint_profile,constraint_violation,feasible" << std::endl;
        for (std::size_t i = 0; i < pop.size(); ++i)
        {
            out << i << "," << pop[i].rank << "," << pop[i].design_fingerprint;
            for (double objective : pop[i].objectives)
                out << "," << objective;
            out << "," << pop[i].compute_only_makespan;
            out << "," << pop[i].bandwidth_makespan_overhead;
            out << "," << pop[i].bandwidth_makespan_overhead /
                std::max(pop[i].objectives[1], 1.0);
            out << "," << pop[i].bandwidth_service_stretch_cycles;
            out << "," << pop[i].bandwidth_queue_delay_cycles;
            out << "," << pop[i].system_bandwidth_queue_delay_cycles;
            out << "," << pop[i].nip_bandwidth_queue_delay_cycles;
            out << "," << pop[i].bandwidth_stretched_layers;
            out << "," << pop[i].bandwidth_delayed_layers;
            out << ",\"";
            for (std::size_t metric = 0; metric < CONSTRAINT_METRIC_COUNT; ++metric)
            {
                if (metric) out << ";";
                out << pop[i].constraint_profile[metric];
            }
            out << "\"," << pop[i].constraint_violation << ","
                << (pop[i].constraint_violation <= 1e-12 ? "true" : "false")
                << std::endl;
        }
    }


    void Moham::OutputEvaluationLedger(std::ofstream &out) const
    {
        out << std::setprecision(17);
        std::vector<EvaluationRecord> records = evaluation_records_;
        std::sort(records.begin(), records.end(),
            [](const auto& left, const auto& right) {
                return left.evaluation_id < right.evaluation_id;
            });
        out << "evaluation_id,evaluation_origin,design_fingerprint,energy_pj,makespan_cycles,area_um2,peak_temperature_c,slo_risk,peak_package_power_w,peak_chiplet_power_w,peak_system_bandwidth,peak_nip_bandwidth,network_completion_cycles,compute_only_makespan_cycles,bandwidth_makespan_overhead_cycles,bandwidth_makespan_overhead_ratio,bandwidth_service_stretch_cycles,bandwidth_queue_delay_cycles,system_bandwidth_queue_delay_cycles,nip_bandwidth_queue_delay_cycles,bandwidth_stretched_layers,bandwidth_delayed_layers,constraint_profile,constraint_violation,feasible" << std::endl;
        for (const auto& record : records)
        {
            out << record.evaluation_id << ","
                << (record.repair ? "repair" : "search") << ","
                << record.design_fingerprint;
            for (double objective : record.objectives)
                out << "," << objective;
            out << "," << record.peak_package_power << ",\"";
            for (std::size_t id = 0; id < record.peak_chiplet_power.size(); ++id)
            {
                if (id) out << ";";
                out << record.peak_chiplet_power[id];
            }
            out << "\"," << record.peak_system_bandwidth << ","
                << record.peak_nip_bandwidth << ",\"";
            for (std::size_t id = 0; id < record.network_completion_times.size(); ++id)
            {
                if (id) out << ";";
                out << record.network_completion_times[id];
            }
            out << "\"," << record.compute_only_makespan << ","
                << record.bandwidth_makespan_overhead << ","
                << record.bandwidth_makespan_overhead /
                    std::max(record.objectives[1], 1.0) << ","
                << record.bandwidth_service_stretch_cycles << ","
                << record.bandwidth_queue_delay_cycles << ","
                << record.system_bandwidth_queue_delay_cycles << ","
                << record.nip_bandwidth_queue_delay_cycles << ","
                << record.bandwidth_stretched_layers << ","
                << record.bandwidth_delayed_layers;
            out << ",\"";
            for (std::size_t metric = 0; metric < CONSTRAINT_METRIC_COUNT; ++metric)
            {
                if (metric) out << ";";
                out << record.constraint_profile[metric];
            }
            out << "\"," << record.constraint_violation << ","
                << (record.constraint_violation <= 1e-12 ? "true" : "false")
                << std::endl;
        }
    }


    void Moham::AppendGenerationInfoToFile(std::ofstream &out, Population &pop, unsigned gen_id)
    {
        out << std::setprecision(17);
        for (Individual &ind : pop)
        {
            out << gen_id << "," << ind.rank << "," << ind.crowding_distance << "," << ind.objectives[0] << "," << ind.objectives[1] << "," << ind.objectives[2] << "," << ind.objectives[3] << "," << ind.objectives[4] << "," << ind.constraint_violation << std::endl;
        }

        out.flush();
    }


    void Moham::AppendEvaluationInfoToFile(std::ofstream &out, Population &pop, unsigned gen_id)
    {
        out << std::setprecision(17);
        for (Individual &ind : pop)
        {
            out << gen_id << "," << ind.rank << "," << ind.crowding_distance;
            for (double objective : ind.objectives)
                out << "," << objective;
            out << "," << ind.peak_package_power;
            out << ",\"";
            for (std::size_t acc_id = 0; acc_id < ind.peak_chiplet_power.size(); ++acc_id)
            {
                if (acc_id) out << ";";
                out << ind.peak_chiplet_power[acc_id];
            }
            out << "\"";
            out << "," << ind.peak_system_bandwidth;
            out << "," << ind.peak_nip_bandwidth;
            out << ",\"";
            for (std::size_t network_id = 0; network_id < ind.network_completion_times.size(); ++network_id)
            {
                if (network_id) out << ";";
                out << ind.network_completion_times[network_id];
            }
            out << "\"";
            out << "," << ind.compute_only_makespan;
            out << "," << ind.bandwidth_makespan_overhead;
            out << "," << ind.bandwidth_makespan_overhead /
                std::max(ind.objectives[1], 1.0);
            out << "," << ind.bandwidth_service_stretch_cycles;
            out << "," << ind.bandwidth_queue_delay_cycles;
            out << "," << ind.system_bandwidth_queue_delay_cycles;
            out << "," << ind.nip_bandwidth_queue_delay_cycles;
            out << "," << ind.bandwidth_stretched_layers;
            out << "," << ind.bandwidth_delayed_layers;
            out << ",\"";
            for (std::size_t metric = 0; metric < CONSTRAINT_METRIC_COUNT; ++metric)
            {
                if (metric) out << ";";
                out << ind.constraint_profile[metric];
            }
            out << "\"";
            out << "," << ind.constraint_violation;
            out << "," << (ind.constraint_violation <= 1e-12 ? "true" : "false");
            out << std::endl;
        }

        out.flush();
    }


    void Moham::AssignBreadthFirstPriorities(Individual& ind) 
    {
        std::size_t num_layers = graph_->NumLayers();
        
        double increase = 1.0 / num_layers;
        double priority = 0.0;

        std::queue<Layer::ID> start_nodes;
        std::vector<int> in_degree(num_layers, 0);

        for (std::size_t l = 0; l < num_layers; l++) 
        {
            in_degree[l] = (*graph_)[l].incoming.size();
            if (in_degree[l] == 0) start_nodes.push(l);
        }   
        
        /*
        while (!start_nodes.empty()) 
        {
            Layer::ID layer_id = start_nodes.front();
            start_nodes.pop();

            ind.genome[layer_id].priority = priority;
            priority += increase;

            for (auto& e : (*graph_)[layer_id].outgoing)
                if (--in_degree[e.layer->id] == 0)
                    start_nodes.push(e.layer->id);
        }
        */

        std::unordered_set<Layer::ID> explored_nodes;
        unsigned debug_num_nodes = 0;

        while (!start_nodes.empty()) 
        {
            Layer::ID layer_id = start_nodes.front();
            start_nodes.pop();

            ind.genome[layer_id].priority = priority;
            priority += increase;

            for (auto& e : (*graph_)[layer_id].outgoing)
            {
                if (explored_nodes.count(e.layer->id) == 0)
                {                  
                    explored_nodes.insert(e.layer->id);
                    start_nodes.push(e.layer->id);
                }
            }
            
            debug_num_nodes++;
        }

        if (debug_num_nodes != num_layers) 
        {
            std::cout << debug_num_nodes << " " << num_layers << std::endl;
            assert(debug_num_nodes == num_layers);
        }
    }


    void Moham::OutputParetoFrontFiles(std::string out_path)
    {
        int max_digits = std::to_string(config_.moham.population_size).length();
        int ind_max_digits = std::to_string(graph_->NumLayers()).length();

        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < parent_population_.size(); i++) 
        {
            auto& ind = parent_population_[i];
            if (ind.rank) continue;

            std::string ind_id = std::to_string(i);
            std::string ind_path = out_path + "/" + std::string(max_digits - ind_id.length(), '0') + ind_id;
            fs::create_directories(ind_path);

            for (std::size_t l = 0; l < graph_->NumLayers(); l++) {
                const Layer& layer = (*graph_)[l];
                const Gene& gene = ind.genome[l];
                ArchInstance& accelerator = ind.accelerators[gene.acc_instance_id];

                auto engine = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].engine;
                auto& mapping = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].genome;
                auto& workload = graph_->GetWorkloads()[layer.workload_id]; 
                auto* sparse_opt = &(*arch_templates_)[accelerator.arch_template_id].sparse_optimizations;
                problem::shape_ = workload.shape; 
                
                engine.Spec(accelerator.arch_specs);
                engine.Evaluate(mapping, workload.workload, sparse_opt);

                std::string l_id = std::to_string(l);
                std::string stats_filename = ind_path + "/moham.stats." + std::string(ind_max_digits - l_id.length(), '0') + l_id + ".txt";
                std::ofstream stats_file(stats_filename);
                stats_file << engine << std::endl;
                stats_file.close();            
            }
        }
    }
    

}
