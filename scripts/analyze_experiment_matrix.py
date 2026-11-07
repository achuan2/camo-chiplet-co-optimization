#!/usr/bin/env python3
"""Compute fair many-objective metrics for a MOHaM experiment matrix."""

import argparse
import csv
import importlib.metadata
import json
import math
import pathlib
import re
import sys

import numpy as np
import yaml
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import rankdata, wilcoxon


OBJECTIVES = [
    "energy_pj",
    "makespan_cycles",
    "area_um2",
    "peak_temperature_c",
    "slo_risk",
]
CONSTRAINTS = [
    "slo",
    "package_power",
    "chiplet_power",
    "temperature",
    "bandwidth",
    "area",
]
FEASIBILITY_EPSILON = 1e-12
BANDWIDTH_DIAGNOSTIC_FIELDS = [
    "compute_only_makespan_cycles",
    "bandwidth_makespan_overhead_cycles",
    "bandwidth_makespan_overhead_ratio",
    "bandwidth_service_stretch_cycles",
    "bandwidth_queue_delay_cycles",
    "system_bandwidth_queue_delay_cycles",
    "nip_bandwidth_queue_delay_cycles",
    "bandwidth_stretched_layers",
    "bandwidth_delayed_layers",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_population_history(path):
    fieldnames = [
        "generation",
        "rank",
        "selection_metric",
        *OBJECTIVES,
        "constraint_violation",
    ]
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, fieldnames=fieldnames))


def is_feasible(row):
    if "feasible" in row:
        return row["feasible"].lower() == "true"
    return float(row["constraint_violation"]) <= FEASIBILITY_EPSILON


def objective_points(rows):
    if not rows:
        return np.empty((0, len(OBJECTIVES)), dtype=float)
    return np.asarray(
        [[float(row[key]) for key in OBJECTIVES] for row in rows], dtype=float
    )


def unique_design_rows(rows):
    """Deduplicate canonical designs when fingerprints are available."""
    if not rows:
        return []
    if "design_fingerprint" not in rows[0]:
        return None
    unique = {}
    for row in rows:
        fingerprint = row["design_fingerprint"]
        if not fingerprint:
            return None
        unique.setdefault(fingerprint, row)
    return list(unique.values())


def common_nondominated_points(rows):
    """Return unique 5-D objective points under one common minimization rule."""
    points = objective_points(rows)
    if not len(points):
        return points
    points = np.unique(points, axis=0)
    indices = NonDominatedSorting().do(points, only_non_dominated_front=True)
    return points[indices]


def low_memory_nondominated_points(points):
    """Return the exact minimization Pareto front without an O(n^2) matrix.

    Lexicographic ordering guarantees that a later unique point cannot
    dominate an earlier one.  It is therefore sufficient to retain an
    incremental archive and test whether any archived point dominates the
    current point.  The method is exact and uses O(n) memory.
    """
    if not len(points):
        return np.empty((0, len(OBJECTIVES)), dtype=float)
    unique = np.unique(points, axis=0)
    order = np.lexsort(tuple(unique[:, column] for column in reversed(
        range(unique.shape[1])
    )))
    archive = np.empty_like(unique)
    archive_size = 0
    for index in order:
        point = unique[index]
        current = archive[:archive_size]
        if archive_size and np.any(
            np.all(current <= point, axis=1)
            & np.any(current < point, axis=1)
        ):
            continue
        archive[archive_size] = point
        archive_size += 1
    return archive[:archive_size].copy()


def nondominated_design_count(rows, nondominated_points):
    if not rows or not len(nondominated_points):
        return 0
    nondominated_keys = {
        tuple(point.tolist()) for point in nondominated_points
    }
    return sum(
        tuple(float(row[key]) for key in OBJECTIVES) in nondominated_keys
        for row in rows
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=pathlib.Path)
    parser.add_argument("scenario")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument(
        "--additional-results-root",
        action="append",
        type=pathlib.Path,
        default=[],
        help="Additional matrix root to include in the same normalization/front.",
    )
    parser.add_argument(
        "--reference-method",
        default="scba-nsga3",
        help="Method compared against every other method in paired tests.",
    )
    parser.add_argument(
        "--require-complete-ledger",
        action="store_true",
        help=(
            "Fail if any run lacks the one-row-per-Evaluate() ledger required "
            "for paper-grade all-search feasibility metrics."
        ),
    )
    parser.add_argument(
        "--allow-method-specific-budgets",
        action="store_true",
        help=(
            "Allow different true evaluation budgets between methods for an "
            "explicit equal-wall-time sensitivity analysis. Each method must "
            "still use one identical budget across all included seeds."
        ),
    )
    parser.add_argument(
        "--strict-wall-clock",
        action="store_true",
        help=(
            "Validate a common graceful wall-clock deadline instead of an "
            "equal true-evaluation budget. Every run must report the same "
            "positive limit and that the limit was reached."
        ),
    )
    return parser.parse_args()


def parse_stats(path):
    stats = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            stats[key] = value
    return stats


def numeric_prefix(value):
    match = re.match(r"[-+0-9.eE]+", value or "")
    return float(match.group(0)) if match else math.nan


def dominates(left, right):
    return np.all(left <= right) and np.any(left < right)


def quartiles(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return math.nan, math.nan, math.nan
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)),
    )


def median_csv_field(rows, field):
    values = [
        float(row[field])
        for row in rows
        if field in row and row[field] not in (None, "")
    ]
    return float(np.median(values)) if values else math.nan


def default_stream_rounding_tolerance(value):
    """Half-unit tolerance for C++'s default six-significant-digit output."""
    value = abs(float(value))
    if value == 0.0 or not math.isfinite(value):
        return 5e-7
    return 0.5 * 10.0 ** (math.floor(math.log10(value)) - 5)


def paired_rank_biserial(oriented_differences):
    """Return a paired rank-biserial effect where positive favors the reference."""
    differences = np.asarray(oriented_differences, dtype=float)
    differences = differences[differences != 0]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences))
    positive = float(np.sum(ranks[differences > 0]))
    negative = float(np.sum(ranks[differences < 0]))
    return (positive - negative) / float(np.sum(ranks))


def holm_adjust(rows):
    by_metric = {}
    for row in rows:
        by_metric.setdefault(row["metric"], []).append(row)
    for metric_rows in by_metric.values():
        ordered = sorted(metric_rows, key=lambda row: row["p_value_one_sided"])
        running_max = 0.0
        count = len(ordered)
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (count - index) * row["p_value_one_sided"])
            running_max = max(running_max, adjusted)
            row["p_value_holm"] = running_max


def parse_constraint_profile(value):
    profile = [float(item) for item in value.split(";")]
    if len(profile) != len(CONSTRAINTS):
        raise ValueError(
            f"Expected {len(CONSTRAINTS)} constraint values, got {profile}"
        )
    return np.asarray(profile, dtype=float)


def main():
    args = parse_args()
    results_root = args.results_root.resolve()
    matrix_roots = [results_root] + [
        path.resolve() for path in args.additional_results_root
    ]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else results_root / "analysis" / args.scenario
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    metrics_paths = []
    for priority, matrix_root in enumerate(matrix_roots):
        metrics_paths.extend(
            (priority, path)
            for path in (matrix_root / "runs" / args.scenario).glob(
                "*/seed-*/moham/moham.pareto_metrics.csv"
            )
        )
    seen_runs = set()
    for _, metrics_path in sorted(metrics_paths, key=lambda item: (item[0], item[1])):
        method = metrics_path.parents[2].name
        seed_name = metrics_path.parents[1].name
        seed = int(seed_name[len("seed-") :] if seed_name.startswith("seed-") else seed_name)
        run_key = (method, seed)
        if run_key in seen_runs:
            continue
        seen_runs.add(run_key)
        stats_path = metrics_path.parent / "moham.stats.txt"
        evaluations_path = metrics_path.parent / "moham.evaluations.csv"
        stats = parse_stats(stats_path)
        run_metadata_path = metrics_path.parent.parent / "run_metadata.json"
        run_metadata = (
            json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if run_metadata_path.exists()
            else {}
        )
        archive_rows = read_csv(metrics_path)
        evaluation_rows = read_csv(evaluations_path)
        feasible_logged_candidates = [
            (index, row)
            for index, row in enumerate(evaluation_rows, start=1)
            if is_feasible(row)
        ]
        first_feasible_logged_candidate = (
            feasible_logged_candidates[0][0]
            if feasible_logged_candidates
            else len(evaluation_rows) + 1
        )
        first_feasible_generation = (
            int(feasible_logged_candidates[0][1]["generation"])
            if feasible_logged_candidates
            else max(int(row["generation"]) for row in evaluation_rows) + 1
        )

        ledger_path = metrics_path.parent / "moham.evaluation_ledger.csv"
        ledger_rows = read_csv(ledger_path) if ledger_path.exists() else []
        ledger_complete = bool(ledger_rows) and len(ledger_rows) == int(
            stats["Evaluations"]
        )
        if args.require_complete_ledger and not ledger_complete:
            raise ValueError(
                f"Incomplete evaluation ledger for {method} seed={seed}: "
                f"{ledger_path}"
            )
        if ledger_complete:
            evaluation_ids = sorted(
                int(row["evaluation_id"]) for row in ledger_rows
            )
            expected_ids = list(range(1, int(stats["Evaluations"]) + 1))
            if evaluation_ids != expected_ids:
                raise ValueError(
                    f"Evaluation IDs are not contiguous for {method} seed={seed}"
                )
            if any(not row["design_fingerprint"] for row in ledger_rows):
                raise ValueError(
                    f"Missing design fingerprint for {method} seed={seed}"
                )
            if "evaluation_origin" not in ledger_rows[0]:
                raise ValueError(
                    f"Missing evaluation origin for {method} seed={seed}"
                )
            repair_rows = [
                row
                for row in ledger_rows
                if row["evaluation_origin"] == "repair"
            ]
            if len(repair_rows) != int(stats["Repair evaluations"]):
                raise ValueError(
                    f"Repair-origin count mismatch for {method} seed={seed}"
                )
            has_bandwidth_diagnostics = all(
                field in ledger_rows[0]
                for field in BANDWIDTH_DIAGNOSTIC_FIELDS
            )
            if has_bandwidth_diagnostics:
                for row in ledger_rows:
                    makespan = float(row["makespan_cycles"])
                    compute_only = float(row["compute_only_makespan_cycles"])
                    overhead = float(row["bandwidth_makespan_overhead_cycles"])
                    ratio = float(row["bandwidth_makespan_overhead_ratio"])
                    expected_overhead = max(0.0, makespan - compute_only)
                    makespan_rounding_tolerance = 1.1 * (
                        default_stream_rounding_tolerance(makespan)
                        + default_stream_rounding_tolerance(compute_only)
                    )
                    overhead_rounding_tolerance = (
                        makespan_rounding_tolerance
                        + 1.1 * default_stream_rounding_tolerance(overhead)
                    )
                    if compute_only > makespan and not math.isclose(
                        compute_only, makespan,
                        rel_tol=5e-5,
                        abs_tol=makespan_rounding_tolerance,
                    ):
                        raise ValueError(
                            f"Compute-only makespan exceeds constrained makespan "
                            f"for {method} seed={seed} evaluation={row['evaluation_id']}"
                        )
                    if not math.isclose(
                        overhead, expected_overhead,
                        rel_tol=5e-5,
                        abs_tol=overhead_rounding_tolerance,
                    ):
                        raise ValueError(
                            f"Bandwidth-overhead mismatch for {method} seed={seed} "
                            f"evaluation={row['evaluation_id']}"
                        )
                    expected_ratio = overhead / max(makespan, 1.0)
                    if not math.isclose(
                        ratio, expected_ratio, rel_tol=5e-4, abs_tol=2e-6
                    ):
                        raise ValueError(
                            f"Bandwidth-overhead ratio mismatch for {method} "
                            f"seed={seed} evaluation={row['evaluation_id']}"
                        )
        else:
            repair_rows = []
            has_bandwidth_diagnostics = False
        feasible_evaluation_rows = (
            [row for row in ledger_rows if is_feasible(row)]
            if ledger_complete
            else []
        )
        ledger_constraint_profiles = (
            np.asarray(
                [
                    parse_constraint_profile(row["constraint_profile"])
                    for row in ledger_rows
                ],
                dtype=float,
            )
            if ledger_complete
            else np.empty((0, len(CONSTRAINTS)), dtype=float)
        )
        unique_feasible_evaluation_rows = unique_design_rows(
            feasible_evaluation_rows
        )
        if ledger_complete and "Feasible evaluations" in stats:
            if len(feasible_evaluation_rows) != int(stats["Feasible evaluations"]):
                raise ValueError(
                    f"Feasible-evaluation count mismatch for {method} seed={seed}"
                )
        if ledger_complete and "Unique feasible designs" in stats:
            if len(unique_feasible_evaluation_rows) != int(
                stats["Unique feasible designs"]
            ):
                raise ValueError(
                    f"Unique-design count mismatch for {method} seed={seed}"
                )

        final_population_path = (
            metrics_path.parent / "moham.final_population_metrics.csv"
        )
        if final_population_path.exists():
            final_population_rows = read_csv(final_population_path)
        else:
            population_rows = read_population_history(
                metrics_path.parent / "moham.populations.csv"
            )
            final_generation = max(
                int(row["generation"]) for row in population_rows
            )
            final_population_rows = [
                row
                for row in population_rows
                if int(row["generation"]) == final_generation
            ]

        final_population_feasible_rows = [
            row for row in final_population_rows if is_feasible(row)
        ]
        unique_final_design_rows = unique_design_rows(
            final_population_feasible_rows
        )
        common_nd_source_rows = (
            unique_final_design_rows
            if unique_final_design_rows is not None
            else final_population_feasible_rows
        )
        final_common_nd_points = common_nondominated_points(
            common_nd_source_rows
        )

        archive_feasible_rows = [
            row for row in archive_rows if is_feasible(row)
        ]
        constraint_profiles = np.asarray(
            [
                parse_constraint_profile(row["constraint_profile"])
                for row in archive_rows
            ],
            dtype=float,
        )
        runs.append(
            {
                "method": method,
                "seed": seed,
                "stats": stats,
                "run_metadata": run_metadata,
                "reported_rank0_count": len(archive_rows),
                "final_archive_feasible_count": len(archive_feasible_rows),
                "final_population_count": len(final_population_rows),
                "final_population_feasible_count": len(
                    final_population_feasible_rows
                ),
                "final_common_nd_feasible_objective_count": len(
                    final_common_nd_points
                ),
                "final_common_nd_feasible_design_count": (
                    nondominated_design_count(
                        common_nd_source_rows, final_common_nd_points
                    )
                    if unique_final_design_rows is not None
                    else math.nan
                ),
                "feasible_evaluations_total": (
                    len(feasible_evaluation_rows)
                    if ledger_complete
                    else math.nan
                ),
                "unique_feasible_designs_total": (
                    len(unique_feasible_evaluation_rows)
                    if unique_feasible_evaluation_rows is not None
                    else math.nan
                ),
                "evaluation_ledger_complete": ledger_complete,
                "ledger_constraint_profiles": ledger_constraint_profiles,
                "ledger_diagnostics": {
                    "median_makespan_cycles_evaluations": median_csv_field(
                        ledger_rows, "makespan_cycles"
                    ),
                    "median_slo_risk_evaluations": median_csv_field(
                        ledger_rows, "slo_risk"
                    ),
                    **{
                        f"median_{field}_evaluations": median_csv_field(
                            ledger_rows, field
                        )
                        for field in BANDWIDTH_DIAGNOSTIC_FIELDS
                    },
                    "median_bandwidth_makespan_overhead_ratio_feasible_evaluations": (
                        median_csv_field(
                            feasible_evaluation_rows,
                            "bandwidth_makespan_overhead_ratio",
                        )
                        if has_bandwidth_diagnostics
                        else math.nan
                    ),
                },
                "repair_evaluation_count_ledger": (
                    len(repair_rows) if ledger_complete else math.nan
                ),
                "repair_feasible_count": (
                    sum(is_feasible(row) for row in repair_rows)
                    if ledger_complete
                    else math.nan
                ),
                "logged_feasible_candidate_count": len(
                    feasible_logged_candidates
                ),
                "logged_evaluation_count": len(evaluation_rows),
                "constraint_profiles": constraint_profiles,
                "points": final_common_nd_points,
                "first_feasible_logged_candidate": first_feasible_logged_candidate,
                "first_feasible_generation": first_feasible_generation,
            }
        )
    if not runs:
        raise ValueError(
            f"No completed runs found for {args.scenario} below {matrix_roots}"
        )

    evaluation_counts = {
        (run["seed"], int(run["stats"]["Evaluations"])) for run in runs
    }
    by_seed = {}
    for seed, count in evaluation_counts:
        by_seed.setdefault(seed, set()).add(count)
    unequal = {seed: counts for seed, counts in by_seed.items() if len(counts) != 1}
    if unequal and not (
        args.allow_method_specific_budgets or args.strict_wall_clock
    ):
        raise ValueError(f"Unequal true evaluation budgets: {unequal}")
    by_method = {}
    for run in runs:
        by_method.setdefault(run["method"], set()).add(
            int(run["stats"]["Evaluations"])
        )
    inconsistent_methods = {
        method: counts for method, counts in by_method.items() if len(counts) != 1
    }
    if inconsistent_methods and not args.strict_wall_clock:
        raise ValueError(
            "Inconsistent true evaluation budgets across seeds: "
            f"{inconsistent_methods}"
        )
    wall_clock_limits = {
        numeric_prefix(run["stats"].get("Wall clock limit", "nan"))
        for run in runs
    }
    wall_clock_reached = {
        run["stats"].get("Wall clock limit reached", "false").lower()
        for run in runs
    }
    if args.strict_wall_clock:
        if len(wall_clock_limits) != 1:
            raise ValueError(
                f"Strict wall-clock runs use different limits: {wall_clock_limits}"
            )
        common_limit = next(iter(wall_clock_limits))
        if not math.isfinite(common_limit) or common_limit <= 0:
            raise ValueError(
                f"Invalid strict wall-clock limit: {common_limit}"
            )
        if wall_clock_reached != {"true"}:
            raise ValueError(
                "Every strict wall-clock run must reach its common deadline; "
                f"observed values: {wall_clock_reached}"
            )

    feasible_point_sets = [run["points"] for run in runs if len(run["points"])]
    if feasible_point_sets:
        all_points = np.vstack(feasible_point_sets)
        reference_raw = low_memory_nondominated_points(all_points)
        reference_keys = {tuple(point.tolist()) for point in reference_raw}
        ideal = np.min(reference_raw, axis=0)
        nadir = np.max(reference_raw, axis=0)
        span = np.where(nadir > ideal, nadir - ideal, 1.0)
        all_normalized = (all_points - ideal) / span
        reference_front = (reference_raw - ideal) / span
        reference_point = np.full(len(OBJECTIVES), 1.1)
        hv_indicator = HV(ref_point=reference_point)
        igd_indicator = IGDPlus(reference_front)
    else:
        ideal = np.full(len(OBJECTIVES), math.nan)
        nadir = np.full(len(OBJECTIVES), math.nan)
        span = np.ones(len(OBJECTIVES))
        all_normalized = np.empty((0, len(OBJECTIVES)))
        reference_front = np.empty((0, len(OBJECTIVES)))
        reference_point = np.full(len(OBJECTIVES), 1.1)
        hv_indicator = None
        igd_indicator = None
        reference_keys = set()

    output_rows = []
    for run in runs:
        normalized = (run["points"] - ideal) / span
        hypervolume = (
            float(hv_indicator(normalized))
            if len(normalized) and hv_indicator is not None
            else (0.0 if hv_indicator is not None else math.nan)
        )
        igd_plus = (
            float(igd_indicator(normalized))
            if len(normalized) and igd_indicator is not None
            else math.inf
        )
        contribution = sum(
            tuple(point.tolist()) in reference_keys
            for point in run["points"]
        )
        profile_median = np.median(run["constraint_profiles"], axis=0)
        profile_max = np.max(run["constraint_profiles"], axis=0)
        output_rows.append(
            {
                "scenario": args.scenario,
                "method": run["method"],
                "seed": run["seed"],
                "evaluations": int(run["stats"]["Evaluations"]),
                "repair_evaluations": int(run["stats"]["Repair evaluations"]),
                "accepted_repair_evaluations": int(
                    run["stats"].get("Accepted repair evaluations", 0)
                ),
                "repair_feasible_rate": (
                    run["repair_feasible_count"]
                    / run["repair_evaluation_count_ledger"]
                    if run["evaluation_ledger_complete"]
                    and run["repair_evaluation_count_ledger"] > 0
                    else math.nan
                ),
                "repair_acceptance_rate": (
                    int(run["stats"].get("Accepted repair evaluations", 0))
                    / run["repair_evaluation_count_ledger"]
                    if run["evaluation_ledger_complete"]
                    and run["repair_evaluation_count_ledger"] > 0
                    else math.nan
                ),
                "adaptive_boundary_parenting": run["stats"].get(
                    "Adaptive boundary parenting", "false"
                ),
                "boundary_archive_mode": run["stats"].get(
                    "Boundary archive mode", "unspecified"
                ),
                "repair_policy": run["stats"].get(
                    "Repair policy", "unspecified"
                ),
                "boundary_parent_probability_max": numeric_prefix(
                    run["stats"].get("Boundary parent probability max", "nan")
                ),
                "scba_parent_selections": numeric_prefix(
                    run["stats"].get("SCBA parent selections", "nan")
                ),
                "scba_boundary_parent_selections": numeric_prefix(
                    run["stats"].get("SCBA boundary parent selections", "nan")
                ),
                "scba_forced_boundary_parent_selections": numeric_prefix(
                    run["stats"].get(
                        "SCBA forced boundary parent selections", "nan"
                    )
                ),
                "scba_boundary_parent_selection_rate": numeric_prefix(
                    run["stats"].get(
                        "SCBA boundary parent selection rate", "nan"
                    )
                ),
                "scba_mean_effective_boundary_probability": numeric_prefix(
                    run["stats"].get(
                        "SCBA mean effective boundary probability", "nan"
                    )
                ),
                "first_feasible_evaluation": int(
                    run["stats"]["First feasible evaluation"]
                ),
                "first_feasible_logged_candidate": run[
                    "first_feasible_logged_candidate"
                ],
                "first_feasible_generation": run["first_feasible_generation"],
                "search_time_s": numeric_prefix(run["stats"]["Search time"]),
                "optimization_wall_time_s": numeric_prefix(
                    run["stats"].get("Optimization wall time", "nan")
                ),
                "wall_clock_limit_s": numeric_prefix(
                    run["stats"].get("Wall clock limit", "nan")
                ),
                "wall_clock_limit_reached": run["stats"].get(
                    "Wall clock limit reached", "false"
                ),
                "wall_clock_drain_overshoot_s": numeric_prefix(
                    run["stats"].get("Wall clock drain overshoot", "nan")
                ),
                "completed_generations": numeric_prefix(
                    run["stats"].get("Completed generations", "nan")
                ),
                "external_process_wall_time_s": float(
                    run["run_metadata"].get(
                        "wall_clock_seconds_external", math.nan
                    )
                ),
                "evaluations_per_optimization_second": (
                    int(run["stats"]["Evaluations"])
                    / numeric_prefix(
                        run["stats"].get("Optimization wall time", "nan")
                    )
                    if numeric_prefix(
                        run["stats"].get("Optimization wall time", "nan")
                    ) > 0
                    else math.nan
                ),
                "evaluation_ledger_complete": run[
                    "evaluation_ledger_complete"
                ],
                "feasible_evaluations_total": run[
                    "feasible_evaluations_total"
                ],
                "feasible_evaluation_rate_total": (
                    run["feasible_evaluations_total"]
                    / int(run["stats"]["Evaluations"])
                    if run["evaluation_ledger_complete"]
                    else math.nan
                ),
                "unique_feasible_designs_total": run[
                    "unique_feasible_designs_total"
                ],
                **run["ledger_diagnostics"],
                "any_violation_evaluation_rate": (
                    1.0
                    - run["feasible_evaluations_total"]
                    / int(run["stats"]["Evaluations"])
                    if run["evaluation_ledger_complete"]
                    else math.nan
                ),
                **{
                    f"{name}_violation_evaluation_rate": (
                        float(
                            np.mean(
                                run["ledger_constraint_profiles"][:, index]
                                > FEASIBILITY_EPSILON
                            )
                        )
                        if run["evaluation_ledger_complete"]
                        else math.nan
                    )
                    for index, name in enumerate(CONSTRAINTS)
                },
                "logged_feasible_candidate_count": run[
                    "logged_feasible_candidate_count"
                ],
                "logged_feasible_candidate_rate": (
                    run["logged_feasible_candidate_count"]
                    / run["logged_evaluation_count"]
                ),
                "final_population_count": run["final_population_count"],
                "final_population_feasible_count": run[
                    "final_population_feasible_count"
                ],
                "final_archive_feasible_count": run[
                    "final_archive_feasible_count"
                ],
                "reported_rank0_count": run["reported_rank0_count"],
                "final_common_nd_feasible_design_count": run[
                    "final_common_nd_feasible_design_count"
                ],
                "final_common_nd_feasible_objective_count": run[
                    "final_common_nd_feasible_objective_count"
                ],
                "slo_satisfaction_rate_rank0": float(
                    np.mean(run["constraint_profiles"][:, 0] <= 1e-12)
                ),
                "median_total_violation_rank0": float(
                    np.median(np.sum(run["constraint_profiles"], axis=1))
                ),
                **{
                    f"median_{name}_violation_rank0": float(profile_median[index])
                    for index, name in enumerate(CONSTRAINTS)
                },
                **{
                    f"max_{name}_violation_rank0": float(profile_max[index])
                    for index, name in enumerate(CONSTRAINTS)
                },
                "hypervolume_5d": hypervolume,
                "igd_plus_5d": igd_plus,
                "joint_front_contribution": contribution,
            }
        )

    output_csv = output_dir / "run_metrics.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    summary_metrics = [
        "hypervolume_5d",
        "igd_plus_5d",
        "feasible_evaluations_total",
        "feasible_evaluation_rate_total",
        "unique_feasible_designs_total",
        "median_makespan_cycles_evaluations",
        "median_slo_risk_evaluations",
        *[
            f"median_{field}_evaluations"
            for field in BANDWIDTH_DIAGNOSTIC_FIELDS
        ],
        "median_bandwidth_makespan_overhead_ratio_feasible_evaluations",
        "repair_feasible_rate",
        "repair_acceptance_rate",
        "scba_boundary_parent_selection_rate",
        "scba_mean_effective_boundary_probability",
        "any_violation_evaluation_rate",
        *[
            f"{name}_violation_evaluation_rate"
            for name in CONSTRAINTS
        ],
        "final_population_feasible_count",
        "final_archive_feasible_count",
        "final_common_nd_feasible_design_count",
        "final_common_nd_feasible_objective_count",
        "logged_feasible_candidate_count",
        "logged_feasible_candidate_rate",
        "slo_satisfaction_rate_rank0",
        "median_total_violation_rank0",
        "first_feasible_logged_candidate",
        "first_feasible_generation",
        "first_feasible_evaluation",
        "search_time_s",
        "optimization_wall_time_s",
        "wall_clock_drain_overshoot_s",
        "external_process_wall_time_s",
        "evaluations_per_optimization_second",
        "joint_front_contribution",
    ]
    summary_rows = []
    methods = sorted({row["method"] for row in output_rows})
    for method in methods:
        method_rows = [row for row in output_rows if row["method"] == method]
        for metric in summary_metrics:
            median, q1, q3 = quartiles([row[metric] for row in method_rows])
            summary_rows.append(
                {
                    "scenario": args.scenario,
                    "method": method,
                    "metric": metric,
                    "n": len(method_rows),
                    "median": median,
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                }
            )
    summary_csv = output_dir / "summary_metrics.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    paired_metrics = {
        "hypervolume_5d": "greater",
        "igd_plus_5d": "less",
        "feasible_evaluations_total": "greater",
        "unique_feasible_designs_total": "greater",
        "final_population_feasible_count": "greater",
        "final_archive_feasible_count": "greater",
        "final_common_nd_feasible_objective_count": "greater",
    }
    reference_rows = {
        row["seed"]: row
        for row in output_rows
        if row["method"] == args.reference_method
    }
    paired_rows = []
    for method in methods:
        if method == args.reference_method:
            continue
        candidate_rows = {
            row["seed"]: row for row in output_rows if row["method"] == method
        }
        for metric, alternative in paired_metrics.items():
            shared_seeds = [
                seed
                for seed in sorted(set(reference_rows) & set(candidate_rows))
                if math.isfinite(float(reference_rows[seed][metric]))
                and math.isfinite(float(candidate_rows[seed][metric]))
            ]
            if not shared_seeds:
                continue
            reference_values = np.asarray(
                [reference_rows[seed][metric] for seed in shared_seeds], dtype=float
            )
            candidate_values = np.asarray(
                [candidate_rows[seed][metric] for seed in shared_seeds], dtype=float
            )
            oriented = (
                reference_values - candidate_values
                if alternative == "greater"
                else candidate_values - reference_values
            )
            if np.all(oriented == 0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(
                        reference_values,
                        candidate_values,
                        alternative=alternative,
                        zero_method="wilcox",
                        mode="auto",
                    ).pvalue
                )
            paired_rows.append(
                {
                    "scenario": args.scenario,
                    "reference_method": args.reference_method,
                    "baseline_method": method,
                    "metric": metric,
                    "alternative": alternative,
                    "n_pairs": len(shared_seeds),
                    "wins": int(np.sum(oriented > 0)),
                    "ties": int(np.sum(oriented == 0)),
                    "losses": int(np.sum(oriented < 0)),
                    "median_oriented_difference": float(np.median(oriented)),
                    "rank_biserial": paired_rank_biserial(oriented),
                    "p_value_one_sided": p_value,
                    "p_value_holm": math.nan,
                }
            )
    holm_adjust(paired_rows)
    paired_csv = output_dir / "paired_tests.csv"
    paired_fieldnames = [
        "scenario",
        "reference_method",
        "baseline_method",
        "metric",
        "alternative",
        "n_pairs",
        "wins",
        "ties",
        "losses",
        "median_oriented_difference",
        "rank_biserial",
        "p_value_one_sided",
        "p_value_holm",
    ]
    with paired_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=paired_fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)

    manifest = {
        "scenario": args.scenario,
        "objectives": OBJECTIVES,
        "normalization": {
            "ideal": ideal.tolist(),
            "nadir": nadir.tolist(),
            "reference_point": reference_point.tolist(),
        },
        "reference_front_points": int(len(reference_front)),
        "metric_definitions": {
            "feasible_evaluations_total": (
                "All actual objective/constraint evaluations that satisfy every "
                "enabled constraint; repeated designs are counted."
            ),
            "unique_feasible_designs_total": (
                "Canonical feasible hardware-mapping-realized-schedule "
                "fingerprints seen across all actual evaluations."
            ),
            "final_population_feasible_count": (
                "Feasible members retained in the final fixed-size population."
            ),
            "final_archive_feasible_count": (
                "Feasible members in the algorithm-reported rank-0 archive."
            ),
            "final_common_nd_feasible_objective_count": (
                "Unique feasible objective vectors from the final population "
                "that are nondominated under the common five-objective rule."
            ),
            "primary_quality_set": (
                "The final common five-objective nondominated feasible points; "
                "this set is used for Hypervolume and IGD+."
            ),
        },
        "legacy_log_warning": (
            "Runs without moham.evaluation_ledger.csv cannot report all-search "
            "feasible evaluations or unique feasible designs."
        ),
        "complete_ledger_required": args.require_complete_ledger,
        "evaluation_budget_mode": (
            "strict-wall-clock"
            if args.strict_wall_clock
            else ("method-specific" if unequal else "equal-evaluations")
        ),
        "wall_clock_check": (
            {
                "common_limit_seconds": next(iter(wall_clock_limits)),
                "all_runs_reached_limit": wall_clock_reached == {"true"},
                "deadline_semantics": (
                    "No new true evaluation starts after the common deadline; "
                    "in-flight evaluations drain before final outputs."
                ),
            }
            if args.strict_wall_clock
            else None
        ),
        "budget_check": (
            {
                method: next(iter(counts))
                for method, counts in sorted(by_method.items())
            }
            if unequal
            else {
                str(seed): next(iter(counts))
                for seed, counts in by_seed.items()
            }
        ),
        "packages": {
            "numpy": importlib.metadata.version("numpy"),
            "pymoo": importlib.metadata.version("pymoo"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "run_metrics_csv": str(output_csv),
        "summary_metrics_csv": str(summary_csv),
        "paired_tests_csv": str(paired_csv),
        "reference_method": args.reference_method,
    }
    with (output_dir / "analysis_manifest.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)

    print(yaml.safe_dump(manifest, sort_keys=False))
    with output_csv.open(encoding="utf-8") as stream:
        print(stream.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
