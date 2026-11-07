# Paper-to-artifact map

| Paper result | Preserved result directory | Inputs |
|---|---|---|
| RQ1 / Figure 2(a): Normal | `results/primary-normal/` | `configs/normal.yaml` |
| RQ1 / Figure 2(a): Temperature-tight | `results/primary-temperature/` | `configs/temperature-tight.yaml` |
| RQ1 / Figure 2(a): Power-tight | `results/primary-power/` | `configs/power-tight.yaml` |
| RQ1 / Figure 2(a): SLO-tight | `results/primary-slo/` | `configs/slo-tight.yaml` |
| RQ2 / Table 5: seven-method ablation | `results/ablation/` | Recorded manifests and fixed-budget snapshot |
| RQ2 / Table 5: equal-cost controls | `results/equal-cost/` | Fixed-budget snapshot and family metrics |
| RQ2: exploratory temperature control | `results/temperature-control/` | Separate five-seed family |
| RQ3 / Figure 2(b) | `results/wallclock/` | Original 900-second manifest and source snapshot |
| RQ4 / Table 6: Holdout-2 | `results/holdout2/` | `configs/holdout2.yaml` |
| RQ4 / Supplementary Table S12 | `results/thermal-audit/` | 60 selected candidate records and calibration report |
| Supplementary Mobile-3 | `results/mobile3/` | Benchmark graphs and family metrics |
| Supplementary Datacenter-4 | `results/datacenter4/`, `results/datacenter4-independent/` | Separate supplementary and independent-seed families |
| Section 5.5 / Table 7 / Tables S16–S23 | `results/ctaea-sensitivity/` | 660 YAMLs, selected-run index, raw Release ZIP |

Each family keeps per-seed metrics, summaries, and paired tests where recorded. Reference fronts and analysis manifests are retained where supplied. The 13 catalog entries contain 2,195 family-level records; analyses can reuse runs, so this is not a count of 2,195 independent optimizer executions.

The approved Figure 1 in `docs/assets/` is an explanatory schematic. Its text and photographs are illustrative inputs, not experimental inference outputs.

