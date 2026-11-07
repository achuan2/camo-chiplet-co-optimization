# Reproducing CAMO

## Verify and reconstruct the included results

From the repository root:

```bash
python scripts/check_artifact.py
python scripts/reproduce_results.py --output outputs/reproduced-results
```

The verifier checks the repository manifest, 256 preserved scientific source/workload payloads, and the result-family catalog. The descriptive reconstruction groups records by scenario, method, and seed, reconstructs medians and quartiles, and checks finite values against preserved summaries.

Outputs:

- `summary_metrics.csv`: regenerated summaries, with an explicit family column.
- `figure2_data.csv`: fixed-evaluation percentage changes and original-RQ3 quality/evaluation medians.
- `verification.json`: compared summary values and any mismatches.

Paired-test outputs remain the recorded `paired_tests.csv` in each family. The lightweight command does not pool or silently re-run statistical families.

```bash
python -m pip install -r requirements-analysis.txt
python scripts/plot_figure2.py --data outputs/reproduced-results/figure2_data.csv \
  --output outputs/figure2
```

The plotting helper reconstructs the numerical panels; journal-specific typography is handled in the manuscript. The SLO-tight IGD+ marker remains nonsignificant.

## Run the optimizer

Build the environment and optimizer using the README commands. Run from the repository root so portable configuration paths resolve correctly.

```bash
./build/moham configs/normal.yaml -o outputs/normal-seed101
./build/moham configs/slo-tight.yaml -o outputs/slo-seed101
./build/moham configs/temperature-tight.yaml -o outputs/temperature-seed101
./build/moham configs/power-tight.yaml -o outputs/power-seed101
./build/moham configs/holdout2.yaml -o outputs/holdout2-seed101
./build/moham configs/normal-900s.yaml -o outputs/normal900-seed101
```

Use three OpenMP threads as in the recorded runs. Mapping pre-search is a separate stage reusable through `search.mapping-cache-dir`. Portable examples set `trust-mapping-cache: false` and generate caches under `outputs/mapping-cache/`.

`configs/recorded/` retains original inputs and manifests. Its cache/working-directory paths describe historical execution; use portable configurations for a fresh checkout. Recorded cache identities are in [Versions](VERSIONS.md).

Algorithm keys are defined in `src/parsing.h`. CAMO uses `algorithm: scba-nsga3` with `objective-mode: slo-risk`; the name is a compatibility identifier. Supplied C-TAEA sensitivity configurations retain their two mating-rule switches. Change settings in a copy and use a new output directory per seed.

## Full raw-data analysis

Install pinned dependencies in Python 3.8 or another compatible isolated environment:

```bash
python -m pip install -r requirements-analysis.txt
```

Verify the matching Release ZIP and extract it at the repository root:

```bash
python scripts/check_raw_release.py /path/to/camo-ctaea-sensitivity-raw-v1.zip
python -m zipfile -e /path/to/camo-ctaea-sensitivity-raw-v1.zip .
python scripts/reanalyze_sensitivity.py --raw data/raw/ctaea-sensitivity \
  --output outputs/sensitivity-reanalysis
```

The entrypoint verifies the 660 selected inputs and prepares a new workspace. It hard-links CSVs when supported, or copies them otherwise. It executes the unchanged scientific-analysis portion of the recorded script, omitting its private handoff-document/ZIP bundler. The original script and plan remain under `analysis/reference/`.

This rebuilds final feasible five-objective nondominated sets, pooled fronts, HV, IGD+, paired tests, Holm families, and recorded bootstrap estimates. Outputs are under `outputs/sensitivity-reanalysis/work/analysis/`. Original CSVs remain unchanged.

A path-and-input check without importing the analysis stack:

```bash
python scripts/reanalyze_sensitivity.py --raw data/raw/ctaea-sensitivity \
  --output outputs/prepared-input-check --prepare-only
```

Choose a new output directory per invocation. The raw index selects 649 original runs and 11 accepted replacements; earlier attempts are not added.

## Analysis conventions

- Five objectives are minimized. Quality sets come from feasible final-population members, with common five-objective nondominance and objective-vector deduplication.
- Each family constructs its own pooled front and ideal/nadir normalization. A constant coordinate uses unit span. HV uses normalized reference `(1.1, 1.1, 1.1, 1.1, 1.1)`.
- Empty final feasible sets retain HV `0` and IGD+ `+inf` in summaries. Finite-pair and zero-difference handling follow the recorded script.
- The seed is the paired statistical unit. Original comparisons and the two mating-rule families keep their own correction groups.
- Feasible evaluations, unique feasible designs, final feasible population size, and final nondominated objective vectors are distinct.
- Timed protocols use admission cutoffs with in-flight completion. Simulated chiplet frequency is separate from host CPU frequency.

Original fixed-budget, original-RQ3, and later sensitivity versions are mapped in [Versions](VERSIONS.md).

