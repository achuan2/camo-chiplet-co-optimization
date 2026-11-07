#!/usr/bin/env python3
"""Derive per-DNN C_ref on one fixed, unmodified Simba chiplet."""

import argparse
import copy
import csv
import pathlib
import subprocess
import sys

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mixed_config", type=pathlib.Path)
    parser.add_argument("mapping_cache_parent", type=pathlib.Path)
    parser.add_argument("results_root", type=pathlib.Path)
    parser.add_argument("--image", default="moham:local")
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument(
        "--writable-reference-cache",
        action="store_true",
        help=(
            "Mount a dedicated reference-only mapping cache read/write so "
            "fixed one-chiplet mappings that are absent from the mixed-workload "
            "cache can be generated. Never point this at a formal shared cache."
        ),
    )
    return parser.parse_args()


def reference_config(base, workload, seed):
    config = copy.deepcopy(base)
    config["workloads"] = [workload]
    architecture = config["architecture"]
    architecture["max-subaccelerators"] = 1
    constraints = architecture.setdefault("constraints", {})
    constraints["package-power-max"] = -1
    constraints["chiplet-power-max"] = -1
    constraints["package-area-max"] = -1
    constraints["enforce-bandwidth"] = False
    constraints.pop("workload-deadlines", None)
    constraints.pop("slo-reference-latencies", None)
    constraints.pop("slo-multiplier", None)
    architecture.setdefault("thermal", {})["enabled"] = False

    search = config["search"]
    search["seed"] = seed
    search["mapping-cache-dir"] = "/mapping-cache/medea"
    scheduler = search["global-scheduler"]
    scheduler["num-generations"] = 0
    scheduler["population-size"] = 2
    scheduler["immigrant-population-size"] = 0
    scheduler["max-per-workload-mappings"] = 0
    scheduler["splitting-mutation-prob"] = 0.0
    scheduler["merging-mutation-prob"] = 0.0
    scheduler["assignment-mutation-prob"] = 0.0
    scheduler["negotiate-arch"] = False
    scheduler["algorithm"] = "nsga2"
    scheduler["objective-mode"] = "legacy"
    return config


def network_name(workload):
    source = workload.get("onnx") or workload.get("benchmark")
    return pathlib.Path(str(source)).stem


def main():
    args = parse_args()
    if args.seed <= 0:
        raise ValueError("The reference seed must be positive.")

    project_root = pathlib.Path(__file__).resolve().parents[1]
    mixed_config = args.mixed_config.resolve()
    mapping_cache_parent = args.mapping_cache_parent.resolve()
    results_root = args.results_root.resolve()
    if args.writable_reference_cache:
        (mapping_cache_parent / "medea").mkdir(parents=True, exist_ok=True)
    elif not (mapping_cache_parent / "medea").is_dir():
        raise ValueError(
            "The read-only mapping cache parent must already contain medea/."
        )
    results_root.mkdir(parents=True, exist_ok=True)
    with mixed_config.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)

    references = []
    for index, workload in enumerate(base["workloads"]):
        name = network_name(workload)
        config = reference_config(base, workload, args.seed)
        config_host = results_root / f"{index:02d}-{name}.yaml"
        with config_host.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)

        run_relative = pathlib.Path("runs") / f"{index:02d}-{name}"
        run_host = results_root / run_relative
        evaluations = run_host / "moham" / "moham.evaluations.csv"
        if not evaluations.exists():
            command = [
                "docker",
                "run",
                "--rm",
                "-e",
                "OMP_NUM_THREADS=8",
                "-v",
                f"{project_root}:/workspaces/moham",
                "-v",
                (
                    f"{mapping_cache_parent}:/mapping-cache"
                    if args.writable_reference_cache
                    else f"{mapping_cache_parent}:/mapping-cache:ro"
                ),
                "-v",
                f"{results_root}:/results",
                "-w",
                "/workspaces/moham/experiments",
                args.image,
                "../build/moham",
                "-c",
                f"/results/{config_host.name}",
                "-o",
                f"/results/{run_relative}",
            ]
            print(f"REFERENCE {name}")
            subprocess.run(command, check=True)

        with evaluations.open(newline="", encoding="utf-8") as stream:
            first = next(csv.DictReader(stream))
        references.append(
            {
                "network": name,
                "reference_latency_cycles": float(first["makespan_cycles"]),
                "architecture": "one fixed unmodified Simba chiplet",
                "seed": args.seed,
            }
        )

    output = {
        "source_config": str(mixed_config),
        "mapping_cache": str(mapping_cache_parent),
        "references": references,
    }
    with (results_root / "slo_references.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    print(yaml.safe_dump(output, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
