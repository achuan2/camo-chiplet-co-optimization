#!/usr/bin/env python3
"""Validate selected MOHaM Pareto schedules with UVA HotSpot 7.0."""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
import statistics
import subprocess


OBJECTIVES = [
    "energy_pj",
    "makespan_cycles",
    "area_um2",
    "peak_temperature_c",
    "slo_risk",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=pathlib.Path)
    parser.add_argument("schedules_csv", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--hotspot-root", required=True, type=pathlib.Path)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument(
        "--candidate-id",
        action="append",
        type=int,
        default=[],
        help="Validate this exact Pareto individual; repeat for multiple IDs.",
    )
    parser.add_argument("--time-step-cycles", type=float, default=100000.0)
    parser.add_argument("--frequency-ghz", type=float, default=1.0)
    parser.add_argument("--ambient-c", type=float, default=25.0)
    parser.add_argument("--chiplet-gap-fraction", type=float, default=0.10)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_candidates(rows, maximum):
    feasible = [row for row in rows if row["feasible"].lower() == "true"]
    if not feasible:
        raise ValueError("HotSpot validation requires feasible Pareto candidates.")

    selected = []

    def add(row):
        candidate = int(row["individual"])
        if candidate not in selected and len(selected) < maximum:
            selected.append(candidate)

    for objective in OBJECTIVES:
        add(min(feasible, key=lambda row: float(row[objective])))
    add(max(feasible, key=lambda row: float(row["peak_temperature_c"])))
    add(min(feasible, key=lambda row: float(row["peak_package_power_w"])))
    add(max(feasible, key=lambda row: float(row["peak_package_power_w"])))

    ideal = {
        objective: min(float(row[objective]) for row in feasible)
        for objective in OBJECTIVES
    }
    nadir = {
        objective: max(float(row[objective]) for row in feasible)
        for objective in OBJECTIVES
    }

    def ideal_distance(row):
        return math.sqrt(
            sum(
                (
                    (float(row[objective]) - ideal[objective])
                    / max(nadir[objective] - ideal[objective], 1e-12)
                )
                ** 2
                for objective in OBJECTIVES
            )
        )

    add(min(feasible, key=ideal_distance))
    ranked = sorted(
        feasible,
        key=lambda row: (
            float(row["peak_temperature_c"]),
            float(row["peak_package_power_w"]),
        ),
    )
    if maximum > len(selected):
        slots = maximum - len(selected)
        for index in range(slots):
            quantile = (index + 1) / (slots + 1)
            add(ranked[round(quantile * (len(ranked) - 1))])
    for row in ranked:
        add(row)
    return selected


def load_selected_schedules(path, selected):
    selected = set(selected)
    schedules = {candidate: [] for candidate in selected}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            candidate = int(row["individual"])
            if candidate in schedules:
                schedules[candidate].append(row)
    missing = [candidate for candidate, rows in schedules.items() if not rows]
    if missing:
        raise ValueError(f"Missing scheduling rows for candidates {missing}")
    return schedules


def build_floorplan(rows, path, gap_fraction):
    accelerators = {}
    for row in rows:
        accelerator = int(row["accelerator"])
        current = (
            float(row["accelerator_area"]),
            int(row["mesh_x"]),
            int(row["mesh_y"]),
        )
        if accelerator in accelerators and accelerators[accelerator] != current:
            raise ValueError(f"Inconsistent accelerator metadata for {accelerator}")
        accelerators[accelerator] = current
    expected = list(range(max(accelerators) + 1))
    if sorted(accelerators) != expected:
        raise ValueError(f"Accelerator IDs must be contiguous: {sorted(accelerators)}")

    side_by_accelerator = {
        accelerator: math.sqrt(max(metadata[0], 1e-12)) * 1e-6
        for accelerator, metadata in accelerators.items()
    }
    pitch = max(side_by_accelerator.values()) * (1.0 + gap_fraction)
    extent_x = 0.0
    extent_y = 0.0
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "# unit-name width height left-x bottom-y; dimensions are meters\n"
        )
        for accelerator in expected:
            _, mesh_x, mesh_y = accelerators[accelerator]
            side = side_by_accelerator[accelerator]
            left = mesh_x * pitch + (pitch - side) / 2.0
            bottom = mesh_y * pitch + (pitch - side) / 2.0
            stream.write(
                f"acc_{accelerator}\t{side:.12g}\t{side:.12g}\t"
                f"{left:.12g}\t{bottom:.12g}\n"
            )
            extent_x = max(extent_x, left + side)
            extent_y = max(extent_y, bottom + side)
    return accelerators, max(extent_x, extent_y)


def build_power_trace(
    rows,
    metric,
    path,
    time_step_cycles,
    frequency_ghz,
):
    accelerator_count = max(int(row["accelerator"]) for row in rows) + 1
    compute_energy = sum(float(row["energy"]) for row in rows)
    total_energy = float(metric["energy_pj"])
    if compute_energy <= 0.0:
        raise ValueError("The selected schedule has non-positive layer energy.")

    has_communication_column = "communication_energy" in rows[0]
    if has_communication_column:
        per_layer_energy = [
            float(row["energy"]) + float(row["communication_energy"])
            for row in rows
        ]
        communication_mode = "exact_per_layer"
    else:
        # Older development outputs do not expose required bandwidth or
        # per-layer communication energy. Preserve exact total energy and
        # state the approximation explicitly in the manifest.
        scale = total_energy / compute_energy
        per_layer_energy = [float(row["energy"]) * scale for row in rows]
        communication_mode = "proportional_total_energy_fallback"

    maximum_end = max(float(row["end_time"]) for row in rows)
    sample_count = max(1, int(math.ceil(maximum_end / time_step_cycles)))
    powers = [
        [0.0 for _ in range(accelerator_count)]
        for _ in range(sample_count)
    ]
    for row, energy_pj in zip(rows, per_layer_energy):
        start = float(row["start_time"])
        end = float(row["end_time"])
        duration = end - start
        if duration <= 0.0:
            continue
        accelerator = int(row["accelerator"])
        constant_power_w = (
            energy_pj / duration * frequency_ghz * 1e-3
        )
        first_sample = max(0, int(math.floor(start / time_step_cycles)))
        final_sample = min(
            sample_count - 1,
            int(math.ceil(end / time_step_cycles)) - 1,
        )
        for sample in range(first_sample, final_sample + 1):
            sample_start = sample * time_step_cycles
            sample_end = sample_start + time_step_cycles
            overlap = max(0.0, min(end, sample_end) - max(start, sample_start))
            powers[sample][accelerator] += (
                constant_power_w * overlap / time_step_cycles
            )

    with path.open("w", encoding="utf-8") as stream:
        stream.write("\t".join(f"acc_{index}" for index in range(accelerator_count)))
        stream.write("\n")
        for sample in powers:
            stream.write("\t".join(f"{power:.12g}" for power in sample))
            stream.write("\n")
    return communication_mode, sample_count


def parse_hotspot_peak(path, ambient_c):
    with path.open(encoding="utf-8") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
        peak_kelvin = ambient_c + 273.15
        sample_count = 0
        for row in reader:
            if not row:
                continue
            values = [float(value) for value in row if value.strip()]
            if values:
                peak_kelvin = max(peak_kelvin, max(values))
                sample_count += 1
    return peak_kelvin - 273.15, sample_count, header


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def pearson(left, right):
    if len(left) < 2:
        return math.nan
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (lhs - left_mean) * (rhs - right_mean)
        for lhs, rhs in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator > 0.0 else math.nan


def main():
    args = parse_args()
    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive.")
    if args.time_step_cycles <= 0.0 or args.frequency_ghz <= 0.0:
        raise ValueError("Time step and frequency must be positive.")

    metrics_path = args.metrics_csv.resolve()
    schedules_path = args.schedules_csv.resolve()
    output_dir = args.output_dir.resolve()
    hotspot_root = args.hotspot_root.resolve()
    hotspot = hotspot_root / "hotspot"
    base_config = hotspot_root / "examples" / "example1" / "example.config"
    if not hotspot.is_file() or not base_config.is_file():
        raise ValueError(f"Invalid built HotSpot root: {hotspot_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    metrics_by_id = {int(row["individual"]): row for row in metrics}
    if args.candidate_id:
        selected = list(dict.fromkeys(args.candidate_id))
        missing = [candidate for candidate in selected if candidate not in metrics_by_id]
        infeasible = [
            candidate
            for candidate in selected
            if candidate in metrics_by_id
            and metrics_by_id[candidate]["feasible"].lower() != "true"
        ]
        if missing:
            raise ValueError(f"Unknown Pareto candidate IDs: {missing}")
        if infeasible:
            raise ValueError(f"HotSpot candidates must be feasible: {infeasible}")
    else:
        selected = choose_candidates(metrics, args.max_candidates)
    schedules = load_selected_schedules(schedules_path, selected)

    result_rows = []
    candidate_manifests = []
    for candidate in selected:
        candidate_dir = output_dir / f"candidate-{candidate}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        floorplan = candidate_dir / "moham.flp"
        power_trace = candidate_dir / "moham.ptrace"
        temperature_trace = candidate_dir / "moham.ttrace"
        log_path = candidate_dir / "hotspot.log"
        copied_config = candidate_dir / "hotspot.config"
        shutil.copyfile(base_config, copied_config)

        accelerators, package_extent = build_floorplan(
            schedules[candidate], floorplan, args.chiplet_gap_fraction
        )
        communication_mode, power_samples = build_power_trace(
            schedules[candidate],
            metrics_by_id[candidate],
            power_trace,
            args.time_step_cycles,
            args.frequency_ghz,
        )
        sampling_seconds = args.time_step_cycles / (
            args.frequency_ghz * 1e9
        )
        ambient_kelvin = args.ambient_c + 273.15
        spreader_side = max(0.03, package_extent * 1.5)
        sink_side = max(0.06, spreader_side * 2.0)
        command = [
            str(hotspot),
            "-c",
            str(copied_config),
            "-f",
            str(floorplan),
            "-p",
            str(power_trace),
            "-model_type",
            "block",
            "-ambient",
            str(ambient_kelvin),
            "-init_temp",
            str(ambient_kelvin),
            "-sampling_intvl",
            str(sampling_seconds),
            "-base_proc_freq",
            str(args.frequency_ghz * 1e9),
            "-s_spreader",
            str(spreader_side),
            "-s_sink",
            str(sink_side),
            "-o",
            str(temperature_trace),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        hotspot_peak, temperature_samples, header = parse_hotspot_peak(
            temperature_trace, args.ambient_c
        )
        compact_peak = float(metrics_by_id[candidate]["peak_temperature_c"])
        result_rows.append(
            {
                "individual": candidate,
                "compact_peak_c": compact_peak,
                "hotspot_peak_c": hotspot_peak,
                "signed_error_c": compact_peak - hotspot_peak,
                "absolute_error_c": abs(compact_peak - hotspot_peak),
                "communication_energy_mode": communication_mode,
                "power_samples": power_samples,
                "temperature_samples": temperature_samples,
                "accelerators": len(accelerators),
            }
        )
        candidate_manifests.append(
            {
                "individual": candidate,
                "command": command,
                "accelerators": {
                    str(key): {
                        "area_um2": value[0],
                        "mesh_x": value[1],
                        "mesh_y": value[2],
                    }
                    for key, value in accelerators.items()
                },
                "hotspot_output_header": header,
                "files": {
                    "floorplan": str(floorplan),
                    "floorplan_sha256": sha256(floorplan),
                    "power_trace": str(power_trace),
                    "power_trace_sha256": sha256(power_trace),
                    "temperature_trace": str(temperature_trace),
                    "temperature_trace_sha256": sha256(temperature_trace),
                },
            }
        )

    output_csv = output_dir / "hotspot_validation.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)

    signed_errors = [row["signed_error_c"] for row in result_rows]
    absolute_errors = [row["absolute_error_c"] for row in result_rows]
    compact_peaks = [row["compact_peak_c"] for row in result_rows]
    hotspot_peaks = [row["hotspot_peak_c"] for row in result_rows]
    summary = {
        "n": len(result_rows),
        "mae_c": statistics.mean(absolute_errors),
        "rmse_c": math.sqrt(
            statistics.mean(error**2 for error in signed_errors)
        ),
        "max_absolute_error_c": max(absolute_errors),
        "max_underprediction_c": max(
            0.0, max(-error for error in signed_errors)
        ),
        "mean_signed_error_c": statistics.mean(signed_errors),
        "pearson": pearson(compact_peaks, hotspot_peaks),
        "spearman": pearson(
            average_ranks(compact_peaks), average_ranks(hotspot_peaks)
        ),
        "recommended_one_sided_safety_margin_c": max(
            0.0, max(-error for error in signed_errors)
        ),
    }
    manifest = {
        "hotspot_root": str(hotspot_root),
        "hotspot_git_commit": subprocess.check_output(
            ["git", "-C", str(hotspot_root), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "hotspot_binary_sha256": sha256(hotspot),
        "metrics_csv": str(metrics_path),
        "metrics_csv_sha256": sha256(metrics_path),
        "schedules_csv": str(schedules_path),
        "schedules_csv_sha256": sha256(schedules_path),
        "frequency_ghz": args.frequency_ghz,
        "time_step_cycles": args.time_step_cycles,
        "sampling_seconds": args.time_step_cycles / (
            args.frequency_ghz * 1e9
        ),
        "ambient_c": args.ambient_c,
        "chiplet_gap_fraction": args.chiplet_gap_fraction,
        "selected_candidates": selected,
        "summary": summary,
        "candidate_manifests": candidate_manifests,
        "results_csv": str(output_csv),
    }
    with (output_dir / "hotspot_validation_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)

    with output_csv.open(encoding="utf-8") as stream:
        print(stream.read())
    print(json.dumps({"summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
