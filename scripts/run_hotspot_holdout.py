#!/usr/bin/env python3
"""Select unique hot feasible Pareto designs and validate them with HotSpot."""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import pathlib
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=pathlib.Path)
    parser.add_argument("scenario")
    parser.add_argument("output_root", type=pathlib.Path)
    parser.add_argument("--hotspot-root", required=True, type=pathlib.Path)
    parser.add_argument("--target-unique", type=int, default=30)
    parser.add_argument("--balanced-per-run", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=5)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.target_unique <= 0 or args.balanced_per_run <= 0 or args.jobs <= 0:
        raise ValueError("Selection counts and jobs must be positive.")
    results_root = args.results_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    script_root = pathlib.Path(__file__).resolve().parent

    runs = []
    for metrics_path in sorted(
        (results_root / "runs" / args.scenario).glob(
            "*/seed-*/moham/moham.pareto_metrics.csv"
        )
    ):
        method = metrics_path.parents[2].name
        seed = metrics_path.parents[1].name
        schedules_path = metrics_path.parent / "moham.pareto_schedulings.csv"
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            feasible = [
                row
                for row in csv.DictReader(stream)
                if row["feasible"].lower() == "true"
            ]
        feasible.sort(
            key=lambda row: float(row["peak_temperature_c"]), reverse=True
        )
        runs.append(
            {
                "method": method,
                "seed": seed,
                "metrics": metrics_path,
                "schedules": schedules_path,
                "candidates": feasible,
                "selected": [],
            }
        )

    selected_fingerprints = set()

    def select(run, row):
        fingerprint = row.get("design_fingerprint", "")
        if not fingerprint:
            raise ValueError(
                f"Missing design fingerprint in {run['metrics']}"
            )
        if fingerprint in selected_fingerprints:
            return False
        selected_fingerprints.add(fingerprint)
        run["selected"].append(row)
        return True

    for position in range(args.balanced_per_run):
        for run in runs:
            if position < len(run["candidates"]):
                select(run, run["candidates"][position])

    remaining = sorted(
        (
            (run, row)
            for run in runs
            for row in run["candidates"]
            if row["design_fingerprint"] not in selected_fingerprints
        ),
        key=lambda item: float(item[1]["peak_temperature_c"]),
        reverse=True,
    )
    for run, row in remaining:
        if len(selected_fingerprints) >= args.target_unique:
            break
        select(run, row)
    if len(selected_fingerprints) < args.target_unique:
        raise ValueError(
            f"Only {len(selected_fingerprints)} unique feasible Pareto designs "
            f"are available; requested {args.target_unique}."
        )

    # Balanced selection can exceed the target. Remove the globally coolest
    # extras while keeping at least one candidate from each non-empty run.
    while len(selected_fingerprints) > args.target_unique:
        removable = [
            (run, row)
            for run in runs
            if len(run["selected"]) > 1
            for row in run["selected"]
        ]
        if not removable:
            break
        run, row = min(
            removable, key=lambda item: float(item[1]["peak_temperature_c"])
        )
        run["selected"].remove(row)
        selected_fingerprints.remove(row["design_fingerprint"])

    tasks = []
    selection = []
    for run in runs:
        if not run["selected"]:
            continue
        output_dir = output_root / f"{run['method']}-{run['seed']}"
        command = [
            sys.executable,
            str(script_root / "validate_hotspot.py"),
            str(run["metrics"]),
            str(run["schedules"]),
            str(output_dir),
            "--hotspot-root",
            str(args.hotspot_root.resolve()),
        ]
        for row in run["selected"]:
            command.extend(["--candidate-id", row["individual"]])
            selection.append(
                {
                    "method": run["method"],
                    "seed": run["seed"],
                    "individual": int(row["individual"]),
                    "design_fingerprint": row["design_fingerprint"],
                    "compact_peak_c": float(row["peak_temperature_c"]),
                }
            )
        tasks.append(command)

    def run_task(command):
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_task, command) for command in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    subprocess.run(
        [
            sys.executable,
            str(script_root / "aggregate_hotspot_validation.py"),
            str(output_root),
            str(output_root / "aggregate"),
        ],
        check=True,
    )
    manifest = {
        "results_root": str(results_root),
        "scenario": args.scenario,
        "target_unique": args.target_unique,
        "selected_unique": len(selected_fingerprints),
        "hotspot_root": str(args.hotspot_root.resolve()),
        "validate_script_sha256": file_sha256(
            script_root / "validate_hotspot.py"
        ),
        "selection": selection,
    }
    with (output_root / "holdout_selection_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

