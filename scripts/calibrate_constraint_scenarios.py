#!/usr/bin/env python3
"""Freeze reproducible Edge-3DNN constraint scenarios from a MOHaM profile."""

import argparse
import csv
import hashlib
import math
import pathlib
import sys

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_config", type=pathlib.Path)
    parser.add_argument("evaluations_csv", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def ceil_to(value, digits):
    scale = 10**digits
    return math.ceil(value * scale - 1e-12) / scale


def write_scenario(base, output_dir, name, alpha, limits, bandwidth):
    config = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
    architecture = config["architecture"]
    constraints = architecture["constraints"]
    constraints["slo-multiplier"] = alpha
    constraints["package-power-max"] = limits["package_power"]
    constraints["chiplet-power-max"] = limits["chiplet_power"]
    constraints["package-area-max"] = limits["area"]
    constraints["enforce-bandwidth"] = True
    validation_margin = float(
        architecture["thermal"].get("validation-safety-margin", 0.0)
    )
    architecture["thermal"]["temperature-max"] = (
        limits["temperature"] + validation_margin
    )
    architecture["system-bandwidth"] = bandwidth["system"]
    architecture["nip-link-bandwidth"] = bandwidth["nip"]
    output = output_dir / f"{name}.yaml"
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return output


def main():
    args = parse_args()
    profile_config = args.profile_config.resolve()
    evaluations_csv = args.evaluations_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with profile_config.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    with evaluations_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The profiling evaluation file is empty.")

    def column(name):
        return [float(row[name]) for row in rows]

    per_individual_chiplet_peak = [
        max(float(value) for value in row["peak_chiplet_power_w"].split(";") if value)
        for row in rows
    ]
    series = {
        "package_power_w": column("peak_package_power_w"),
        "chiplet_power_w": per_individual_chiplet_peak,
        "temperature_c": column("peak_temperature_c"),
        "area_um2": column("area_um2"),
    }
    quantiles = {
        metric: {
            "q50": quantile(values, 0.50),
            "q75": quantile(values, 0.75),
            "q90": quantile(values, 0.90),
        }
        for metric, values in series.items()
    }

    normal_limits = {
        "package_power": ceil_to(quantiles["package_power_w"]["q90"], 2),
        "chiplet_power": ceil_to(quantiles["chiplet_power_w"]["q90"], 2),
        "temperature": ceil_to(quantiles["temperature_c"]["q90"], 2),
        "area": ceil_to(quantiles["area_um2"]["q90"] / 100000.0, 0) * 100000.0,
    }
    power_tight_limits = dict(normal_limits)
    power_tight_limits.update(
        {
            "package_power": ceil_to(quantiles["package_power_w"]["q50"], 2),
            "chiplet_power": ceil_to(quantiles["chiplet_power_w"]["q75"], 2),
        }
    )
    thermal_tight_limits = dict(normal_limits)
    thermal_tight_limits["temperature"] = ceil_to(
        quantiles["temperature_c"]["q50"], 2
    )

    normal_bandwidth = {
        "system": float(base["architecture"]["system-bandwidth"]),
        "nip": float(base["architecture"]["nip-link-bandwidth"]),
    }
    # Stress both shared DRAM capacity and deterministic XY-routed NoP links.
    tight_bandwidth = {"system": 8.0, "nip": 2.0}
    scenario_specs = {
        "edge3_normal_alpha600": (6.0, normal_limits, normal_bandwidth),
        "edge3_slo_tight_alpha450": (4.5, normal_limits, normal_bandwidth),
        "edge3_power_tight_alpha600": (
            6.0,
            power_tight_limits,
            normal_bandwidth,
        ),
        "edge3_thermal_tight_alpha600": (
            6.0,
            thermal_tight_limits,
            normal_bandwidth,
        ),
        "edge3_bandwidth_tight_alpha600": (
            6.0,
            normal_limits,
            tight_bandwidth,
        ),
    }

    generated = {}
    for name, (alpha, limits, bandwidth) in scenario_specs.items():
        generated[name] = str(
            write_scenario(base, output_dir, name, alpha, limits, bandwidth)
        )

    report = {
        "policy": (
            "Thresholds were frozen from the original MOHaM seed-101 profile "
            "before constrained method comparison. Normal limits use q90; "
            "power-tight uses package q50 and chiplet q75; thermal-tight uses "
            "temperature q50."
        ),
        "profile_config": str(profile_config),
        "profile_config_sha256": file_sha256(profile_config),
        "evaluations_csv": str(evaluations_csv),
        "evaluations_csv_sha256": file_sha256(evaluations_csv),
        "evaluation_rows": len(rows),
        "quantiles": quantiles,
        "normal_limits": normal_limits,
        "thermal_validation_safety_margin_c": float(
            base["architecture"]["thermal"].get(
                "validation-safety-margin", 0.0
            )
        ),
        "generated_scenarios": generated,
    }
    report_path = output_dir / "calibration_report.yaml"
    with report_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
