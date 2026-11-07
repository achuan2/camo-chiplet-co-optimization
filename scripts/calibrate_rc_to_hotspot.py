#!/usr/bin/env python3
"""Fit MOHaM's compact RC parameters to generated HotSpot validation traces."""

import argparse
import csv
import hashlib
import json
import math
import pathlib

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.stats import spearmanr


DEFAULT_PARAMETERS = np.asarray(
    [15.0, 50.0, 0.005, 0.1, 140.4], dtype=float
)
PARAMETER_NAMES = [
    "self_resistance_k_mm2_per_w",
    "neighbor_resistance_k_per_w",
    "capacitance_j_per_k_mm2",
    "package_resistance_k_per_w",
    "package_capacitance_j_per_k",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_root", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    parser.add_argument("--decimation", type=int, default=10)
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument(
        "--skip-cross-validation",
        action="store_true",
        help="Fit all cases only; useful after a separate CV run is archived.",
    )
    parser.add_argument(
        "--initial-parameters",
        nargs=len(PARAMETER_NAMES),
        type=float,
        metavar=tuple(PARAMETER_NAMES),
        help="Use one explicit physical-parameter starting point.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(root, decimation):
    cases = []
    seen_inputs = set()
    for manifest_path in sorted(root.glob("*/hotspot_validation_manifest.json")):
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        method = manifest_path.parent.name
        with pathlib.Path(manifest["results_csv"]).open(
            newline="", encoding="utf-8"
        ) as stream:
            results = {
                int(row["individual"]): row for row in csv.DictReader(stream)
            }
        for candidate in manifest["candidate_manifests"]:
            candidate_id = int(candidate["individual"])
            floorplan_hash = candidate["files"]["floorplan_sha256"]
            power_hash = candidate["files"]["power_trace_sha256"]
            input_key = (floorplan_hash, power_hash)
            if input_key in seen_inputs:
                continue
            seen_inputs.add(input_key)

            power_path = pathlib.Path(candidate["files"]["power_trace"])
            powers = np.loadtxt(power_path, delimiter="\t", skiprows=1)
            if powers.ndim == 1:
                powers = powers.reshape(-1, 1)
            if decimation > 1:
                remainder = len(powers) % decimation
                if remainder:
                    powers = np.pad(
                        powers,
                        ((0, decimation - remainder), (0, 0)),
                        mode="constant",
                    )
                powers = powers.reshape(
                    -1, decimation, powers.shape[1]
                ).mean(axis=1)

            accelerators = candidate["accelerators"]
            count = len(accelerators)
            area_mm2 = np.zeros(count, dtype=float)
            coordinates = np.zeros((count, 2), dtype=int)
            for identifier, metadata in accelerators.items():
                index = int(identifier)
                area_mm2[index] = float(metadata["area_um2"]) * 1e-6
                coordinates[index] = [
                    int(metadata["mesh_x"]),
                    int(metadata["mesh_y"]),
                ]

            distance_reciprocal = np.zeros((count, count), dtype=float)
            for left in range(count):
                for right in range(count):
                    if left == right:
                        continue
                    distance = np.abs(
                        coordinates[left] - coordinates[right]
                    ).sum()
                    if distance > 0:
                        distance_reciprocal[left, right] = 1.0 / distance
            laplacian = np.diag(distance_reciprocal.sum(axis=1)) - (
                distance_reciprocal
            )
            result = results[candidate_id]
            cases.append(
                {
                    "method": method,
                    "individual": candidate_id,
                    "power_trace": str(power_path),
                    "power_trace_sha256": sha256(power_path),
                    "powers": powers,
                    "area_mm2": area_mm2,
                    "laplacian": laplacian,
                    "dt_seconds": (
                        float(manifest["sampling_seconds"]) * decimation
                    ),
                    "ambient_c": float(manifest["ambient_c"]),
                    "hotspot_peak_c": float(result["hotspot_peak_c"]),
                    "reported_compact_peak_c": float(
                        result["compact_peak_c"]
                    ),
                }
            )
    if not cases:
        raise ValueError(f"No validation manifests found below {root}")
    return cases


def simulate_peak(case, parameters):
    (
        self_resistance,
        neighbor_resistance,
        capacitance_per_area,
        package_resistance,
        package_capacitance,
    ) = parameters
    area = case["area_mm2"]
    capacity = np.maximum(capacitance_per_area * area, 1e-12)
    self_conductance = area / self_resistance
    temperature = np.full(len(area), case["ambient_c"], dtype=float)
    package_temperature = case["ambient_c"]
    peak = case["ambient_c"]
    for powers in case["powers"]:
        heat_to_package = self_conductance * (
            temperature - package_temperature
        )
        heat_between_chiplets = (
            case["laplacian"].dot(temperature) / neighbor_resistance
        )
        temperature = temperature + case["dt_seconds"] * (
            powers - heat_to_package - heat_between_chiplets
        ) / capacity
        package_temperature = package_temperature + case["dt_seconds"] * (
            np.sum(heat_to_package)
            - (package_temperature - case["ambient_c"]) / package_resistance
        ) / package_capacitance
        if not np.all(np.isfinite(temperature)) or np.max(
            np.abs(temperature)
        ) > 1e6 or not math.isfinite(package_temperature):
            return 1e6
        peak = max(peak, float(np.max(temperature)))
    return peak


def fit(cases, max_nfev, initial_parameters=None):
    targets = np.asarray([case["hotspot_peak_c"] for case in cases])
    initial_parameters = (
        DEFAULT_PARAMETERS
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=float)
    )

    def residual(log_parameters):
        parameters = np.exp(log_parameters)
        predictions = np.asarray(
            [simulate_peak(case, parameters) for case in cases]
        )
        return np.clip(predictions - targets, -1e4, 1e4)

    lower = np.log([0.1, 0.1, 1e-5, 0.001, 1e-3])
    upper = np.log([1000.0, 10000.0, 10.0, 100.0, 10000.0])
    result = least_squares(
        residual,
        np.log(initial_parameters),
        bounds=(lower, upper),
        max_nfev=max_nfev,
        x_scale="jac",
    )
    return np.exp(result.x), result


def metrics(cases, parameters):
    targets = np.asarray([case["hotspot_peak_c"] for case in cases])
    predictions = np.asarray(
        [simulate_peak(case, parameters) for case in cases]
    )
    errors = predictions - targets
    if len(cases) > 1:
        correlation = float(spearmanr(targets, predictions).statistic)
    else:
        correlation = math.nan
    return {
        "n": len(cases),
        "mae_c": float(np.mean(np.abs(errors))),
        "rmse_c": float(np.sqrt(np.mean(errors**2))),
        "max_absolute_error_c": float(np.max(np.abs(errors))),
        "mean_signed_error_c": float(np.mean(errors)),
        "spearman": correlation,
    }, predictions


def main():
    args = parse_args()
    if args.decimation <= 0 or args.max_nfev <= 0:
        raise ValueError("Decimation and max-nfev must be positive.")
    validation_root = args.validation_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(validation_root, args.decimation)

    starts = (
        [np.asarray(args.initial_parameters, dtype=float)]
        if args.initial_parameters
        else [
            DEFAULT_PARAMETERS,
            np.asarray([8.0, 5000.0, 0.0002, 0.6, 0.3]),
            np.asarray([15.0, 5000.0, 0.0004, 0.5, 0.5]),
        ]
    )
    fits = [
        fit(cases, args.max_nfev, initial_parameters=start)
        for start in starts
    ]
    calibrated, optimization = min(
        fits, key=lambda item: item[1].cost
    )
    default_metrics, default_predictions = metrics(
        cases, DEFAULT_PARAMETERS
    )
    calibrated_metrics, calibrated_predictions = metrics(cases, calibrated)

    cross_validation = []
    methods = sorted({case["method"] for case in cases})
    if not args.skip_cross_validation:
        for held_out in methods:
            train = [case for case in cases if case["method"] != held_out]
            test = [case for case in cases if case["method"] == held_out]
            if not train or not test:
                continue
            candidate_fits = [
                fit(
                    train,
                    args.max_nfev,
                    initial_parameters=DEFAULT_PARAMETERS,
                ),
                fit(
                    train,
                    args.max_nfev,
                    initial_parameters=calibrated,
                ),
            ]
            parameters, result = min(
                candidate_fits, key=lambda item: item[1].cost
            )
            train_metrics, _ = metrics(train, parameters)
            test_metrics, _ = metrics(test, parameters)
            cross_validation.append(
                {
                    "held_out_method": held_out,
                    "parameters": {
                        name: float(value)
                        for name, value in zip(PARAMETER_NAMES, parameters)
                    },
                    "optimizer_success": bool(result.success),
                    "train": train_metrics,
                    "test": test_metrics,
                }
            )

    result_rows = []
    for case, default_prediction, calibrated_prediction in zip(
        cases, default_predictions, calibrated_predictions
    ):
        result_rows.append(
            {
                "method": case["method"],
                "individual": case["individual"],
                "chiplets": len(case["area_mm2"]),
                "reported_compact_peak_c": case[
                    "reported_compact_peak_c"
                ],
                "resampled_default_peak_c": default_prediction,
                "calibrated_peak_c": calibrated_prediction,
                "hotspot_peak_c": case["hotspot_peak_c"],
                "calibrated_signed_error_c": (
                    calibrated_prediction - case["hotspot_peak_c"]
                ),
                "power_trace_sha256": case["power_trace_sha256"],
            }
        )
    results_csv = output_dir / "rc_calibration_cases.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)

    report = {
        "validation_root": str(validation_root),
        "decimation": args.decimation,
        "cross_validation_skipped": args.skip_cross_validation,
        "case_count": len(cases),
        "methods": methods,
        "default_parameters": {
            name: float(value)
            for name, value in zip(PARAMETER_NAMES, DEFAULT_PARAMETERS)
        },
        "calibrated_parameters": {
            name: float(value)
            for name, value in zip(PARAMETER_NAMES, calibrated)
        },
        "optimizer": {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "function_evaluations": int(optimization.nfev),
            "cost": float(optimization.cost),
            "multi_start_count": len(starts),
        },
        "default_metrics": default_metrics,
        "calibrated_metrics": calibrated_metrics,
        "leave_one_method_out": cross_validation,
        "cases_csv": str(results_csv),
    }
    report_path = output_dir / "rc_calibration.yaml"
    with report_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
