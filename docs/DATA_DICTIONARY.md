# Data dictionary

## Objectives and physical quantities

| Field | Meaning | Unit |
|---|---|---|
| `energy_pj` | Scheduled compute and NoP communication energy | pJ |
| `makespan_cycles` | Batch makespan | simulated cycles |
| `area_um2` | Aggregate chiplet silicon area | µm² |
| `peak_temperature_c` | Peak compact-RC chiplet temperature | °C |
| `slo_risk` | Maximum completion/deadline ratio | dimensionless |
| `peak_package_power_w`, `peak_chiplet_power_w` | Peak scheduled dynamic power | W |
| `constraint_violation` | Stored aggregate normalized violation | dimensionless |
| `constraint_profile` | Semicolon-separated slots | SLO, package power, chiplet power, temperature, bandwidth residual, area |

The effective profile has five categories; bandwidth is scheduler-enforced. The implementation retains its numerical bandwidth residual in CV and profile distance. A true `feasible` field supplies the recorded legality/feasibility decision.

## Search and quality metrics

| Field | Definition |
|---|---|
| `evaluations` | True evaluator calls, including evaluated rejected repairs |
| `feasible_evaluations_total` | Feasible calls across the history, including repetitions |
| `unique_feasible_designs_total` | Feasible history deduplicated by fingerprint |
| `final_population_feasible_count` | Feasible final-population members |
| `final_common_nd_feasible_objective_count` | Distinct feasible five-objective nondominated final vectors |
| `hypervolume_5d` | HV under the family's shared normalized reference |
| `igd_plus_5d` | IGD+ under that reference; smaller is better |
| `first_feasible_evaluation` | Recorded zero-based evaluation ID; the paper adds one where stated |

Machine-local `raw_output` paths are omitted from public processed metrics. Scientific values and seed/method keys are preserved; `provenance/files.json` records source/public hashes.

## Method identifiers

`camo`, `cpar-nsga3`, and `scba-nsga3` identify CAMO in recorded families. `scba-nsga3` is also the configuration key; these are not additional baselines.

Sensitivity V00 uses CA-internal second-parent probability and archive redraw for an infeasible sampled pair. V10 changes to combined-archive contribution; V01 changes the infeasible-pair choice; V11 changes both. Remaining conventions are shared.

## Statistics

`median`, `q1`, `q3`, and `iqr` describe seed-level distributions; IQR is `q3 − q1`. Fixed-budget displays normally use median [IQR]; timed displays can use median [Q1, Q3]. Empty final sets retain HV `0` and IGD+ `inf`. Undefined finite IQRs can appear as `nan`/N/A.

Original paired-test files use `p_value_one_sided` and `p_value_holm`; sensitivity files use `p_raw`, `p_holm`, and an explicit family. Alternative, finite-pair count, zero differences, effect sign, and correction family remain in each file.

## Timing

In sensitivity compatibility metrics, `optimizer_time_s` is the last evaluator completion time, not optimizer termination. `TIMING_FIELD_AUDIT.csv` contains separately recorded `optimization_wall_time_record` and search time. `external_time_s` includes setup and serialization. The 900-second cutoff governs admission; admitted calls complete normally.

