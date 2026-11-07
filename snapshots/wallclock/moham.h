#ifndef MOHAM_MOHAM_H_
#define MOHAM_MOHAM_H_

#include "nsga.h"

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include "config.h"
#include "graph.h"
#include "archtemplate.h"
#include "medea.h"

namespace moham {

    // Separating the violations preserves the reason an individual is
    // infeasible; this is the state used by the SCBA boundary archive.
    enum ConstraintMetric : std::size_t {
        SLO_VIOLATION,
        PACKAGE_POWER_VIOLATION,
        CHIPLET_POWER_VIOLATION,
        TEMPERATURE_VIOLATION,
        BANDWIDTH_VIOLATION,
        AREA_VIOLATION,
        CONSTRAINT_METRIC_COUNT
    };

    enum class BottleneckKind {
        SLO, PACKAGE_POWER, CHIPLET_POWER, TEMPERATURE, BANDWIDTH, AREA
    };

    struct MemoryInterface {
        typedef std::size_t ID;

        ID id;
        std::pair<int,int> mesh_coordinates;
    };


    struct ArchInstance 
    {
        typedef std::size_t ID;

        ID id;
        ArchTemplate::ID arch_template_id;
        MinimalArchSpecs arch;
        timeloop::EngineSpecs arch_specs;
        double area;
        bool need_evaluation;
        unsigned num_assigned_layers;

        std::pair<int,int> mesh_coordinates;
        MemoryInterface::ID memory_interface_id;
    };


    class Moham : public NSGA
    {
    public:
        struct LayerEval {
            double energy, cycles, area, required_bandwidth;
            double start_time, end_time;
            bool need_evaluation;
        };

        struct Gene {
            double priority;
            Medea::Individual::ID mapping_id;
            ArchInstance::ID acc_instance_id;
        };

        struct BottleneckSegment {
            double start_time, end_time;
            ArchInstance::ID acc;
            Layer::ID layer_id;
        };

        struct BottleneckAttribution {
            BottleneckKind kind;
            Layer::ID layer_id;
            ArchInstance::ID accelerator_id;
            double contribution;
        };

        struct Individual : NSGA::Individual<5>
        {
            std::vector<Gene> genome;

            std::vector<LayerEval> layer_evals;

            std::vector<ArchInstance> accelerators;
            std::vector<MemoryInterface> memory_interfaces;
            std::pair<int,int> mesh_dimensions;

            std::vector<Layer::ID> toposort;
            std::vector<BottleneckSegment> nip_bottlenecks;
            std::vector<BottleneckSegment> memory_bottlenecks;

            double peak_package_power = 0.0;
            std::vector<double> peak_chiplet_power;
            double peak_temperature = 0.0;
            double peak_system_bandwidth = 0.0;
            double peak_nip_bandwidth = 0.0;
            // Diagnostics only: these values do not participate in objectives,
            // constraints, selection, or repair decisions.
            double compute_only_makespan = 0.0;
            double bandwidth_makespan_overhead = 0.0;
            double bandwidth_service_stretch_cycles = 0.0;
            double bandwidth_queue_delay_cycles = 0.0;
            double system_bandwidth_queue_delay_cycles = 0.0;
            double nip_bandwidth_queue_delay_cycles = 0.0;
            std::size_t bandwidth_stretched_layers = 0;
            std::size_t bandwidth_delayed_layers = 0;
            // Completion times are indexed by input DNN/network, not by
            // MOHaM's deduplicated Timeloop layer-workload IDs.
            std::vector<double> network_completion_times;
            std::array<double, CONSTRAINT_METRIC_COUNT> constraint_profile{};
            std::vector<BottleneckAttribution> bottleneck_ledger;
            unsigned long long evaluation_id = 0;
            std::string design_fingerprint;
        };

        typedef std::vector<Individual> Population;

    private:

        Config config_;
        Graph* graph_;
        std::vector<ArchTemplate>* arch_templates_;
        std::vector<std::unordered_map<ArchTemplate::ID,Medea::Population>> workload_mappings_;

        std::vector<std::unordered_map<ArchTemplate::ID,std::array<std::pair<double,double>,3>>> workload_mappings_objectives_bounds_;
        std::vector<std::unordered_map<ArchTemplate::ID,std::vector<std::array<double,3>>>> workload_mappings_normalized_objectives_;

        Population population_, parent_population_, immigrant_population_, merged_population_;
        Population feasible_archive_, boundary_archive_;
        Population ctaea_convergence_archive_, ctaea_diversity_archive_;
        double ctaea_first_parent_from_convergence_ = 1.0;
        double ctaea_second_parent_convergence_probability_ = 1.0;

        std::vector<std::string> network_names_;
        std::vector<std::size_t> layer_network_ids_;
        std::atomic<unsigned long long> evaluation_count_{0};
        std::atomic<unsigned long long> first_feasible_evaluation_{0};
        std::atomic<unsigned> repair_evaluation_budget_{0};
        std::atomic<unsigned long long> repair_evaluation_count_{0};
        double effective_boundary_parent_probability_ = 0.0;
        std::atomic<unsigned long long> scba_parent_selection_count_{0};
        std::atomic<unsigned long long> scba_boundary_parent_selection_count_{0};
        std::atomic<unsigned long long> scba_forced_boundary_parent_selection_count_{0};
        std::atomic<unsigned long long> scba_effective_probability_sum_ppm_{0};

        struct EvaluationRecord {
            unsigned long long evaluation_id = 0;
            std::string design_fingerprint;
            bool repair = false;
            std::array<double, 5> objectives{};
            double peak_package_power = 0.0;
            std::vector<double> peak_chiplet_power;
            double peak_system_bandwidth = 0.0;
            double peak_nip_bandwidth = 0.0;
            double compute_only_makespan = 0.0;
            double bandwidth_makespan_overhead = 0.0;
            double bandwidth_service_stretch_cycles = 0.0;
            double bandwidth_queue_delay_cycles = 0.0;
            double system_bandwidth_queue_delay_cycles = 0.0;
            double nip_bandwidth_queue_delay_cycles = 0.0;
            std::size_t bandwidth_stretched_layers = 0;
            std::size_t bandwidth_delayed_layers = 0;
            std::vector<double> network_completion_times;
            std::array<double, CONSTRAINT_METRIC_COUNT> constraint_profile{};
            double constraint_violation = 0.0;
        };
        std::mutex evaluation_records_mutex_;
        std::vector<EvaluationRecord> evaluation_records_;
        std::atomic<unsigned long long> accepted_repair_evaluation_count_{0};
        bool wall_clock_limit_active_ = false;
        std::chrono::steady_clock::time_point wall_clock_deadline_;
        std::atomic<bool> wall_clock_limit_reached_{false};

        std::vector<std::mt19937_64> rng_;
        std::uniform_real_distribution<double> uni_distribution_;

    public:

        Moham(
            Config config, 
            Graph* graph, 
            std::vector<ArchTemplate>* arch_templates, 
            std::vector<std::unordered_map<ArchTemplate::ID,Medea::Population>> workload_mappings
        );

        void Run();

        void RunRandom();
        
    private:

        bool WallClockExpired();

        Individual RandomIndividual();

        double RandomPopulation(Population& population);

        void InjectHeuristicallyGoodIndividuals(Population& population);
        
        std::vector<Layer::ID> GenerateTopologicalSorting(const Individual& individual);

        void Evaluate(Individual& individual, bool repair = false);
        std::string DesignFingerprint(const Individual& individual) const;
        void RecordEvaluation(const Individual& individual, bool repair);

        bool CheckIndividualValidity(Individual& individual, bool do_assert = true) const;

        void CalculatePackageMesh(Individual& individual);
        double EvaluateMakespan(Individual& individual, const std::vector<Layer::ID>& topological_sorting);
        double EvaluateLayerCommunicationEnergy(const Individual& individual, Layer::ID layer_id) const;
        double EvaluateNipEnergy(Individual& individual);
        void EvaluatePhysicalMetrics(Individual& individual);
        void EvaluateConstraints(Individual& individual);
        void BuildBottleneckLedger(Individual& individual) const;
        double EvaluatePeakTemperature(const Individual& individual) const;
        double EvaluatePeakBandwidth(const Individual& individual, bool per_memory_interface) const;

        bool ConstraintProfileDominates(const Individual& a, const Individual& b) const;
        double ConstraintProfileDistance(const Individual& a, const Individual& b) const;
        void UpdateSCBAArchives(const Population& candidates);
        void UpdateReferenceDirectionBias();
        void UpdateSCBABoundaryPolicy();
        const Individual& SelectSCBAParent();
        void UpdateCTAEAArchives(const Population& candidates);
        void UpdateCTAEAMatingProbabilities();
        const Individual& SelectCTAEAParent(bool first_parent);
        const Individual& CTAEAMatingSelection(const Population& archive);
        Population UpdateCTAEAConvergence(
            const Population& candidates,
            const std::vector<std::vector<double>>& directions);
        Population UpdateCTAEADiversity(
            const Population& candidates,
            const std::vector<std::vector<double>>& directions);
        std::vector<std::size_t> AssociateCTAEARegions(
            const Population& population,
            const std::vector<std::vector<double>>& directions) const;
        bool ApplyBottleneckGuidedRepair(Individual& individual, std::mt19937_64& rng);
        bool MoveLayerToCompatibleAccelerator(Individual& individual, Layer::ID layer_id, bool prefer_coolest) const;
        bool MoveLayerForBandwidthRelief(Individual& individual, Layer::ID layer_id) const;

        void NegotiateArchitectures(Individual &individual);

        void Crossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b);
        void PriorityCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng);
        void SubAcceleratorCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng);
        void MappingCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng);
        
        void Mutation(Individual& ind);
        void SplittingMutation(Individual& ind, std::mt19937_64& rng);
        void MergingMutation(Individual& ind, std::mt19937_64& rng);
        void TemplateMutation(Individual& ind, std::mt19937_64& rng);
        void PositionMutation(Individual& ind, std::mt19937_64& rng);
        void PriorityMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng);
        void MappingMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng);
        void AssignmentMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng);

        void RemoveEmptySubAccelerators(Individual& ind) const;

        Medea::Individual::ID ConvertMappingToTemplate(Workload::ID workload_id, Medea::Individual::ID mapping_id, ArchTemplate::ID start_template, ArchTemplate::ID destination_template) const;

        std::size_t Tournament(const Population& pop);
        
        void OutputParetoScheduling(std::ofstream &out, const Population& pop);
        void OutputParetoNipBottlenecks(std::ofstream &out, const Population& pop);
        void OutputParetoMetrics(std::ofstream &out, const Population& pop);
        void OutputFinalPopulationMetrics(std::ofstream &out, const Population& pop);
        void OutputEvaluationLedger(std::ofstream &out) const;

        void AppendGenerationInfoToFile(std::ofstream &out, Population &pop, unsigned gen_id);
        void AppendEvaluationInfoToFile(std::ofstream &out, Population &pop, unsigned gen_id);

        void AssignBreadthFirstPriorities(Individual& ind);

        void OutputParetoFrontFiles(std::string out_path);
    };
}


#endif // MOHAM_MOHAM_H_
