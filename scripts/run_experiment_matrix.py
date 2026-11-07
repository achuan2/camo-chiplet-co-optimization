#!/usr/bin/env python3
"""Run fair MOHaM scheduler comparisons with a shared layer-mapping cache."""

import argparse
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
import pathlib
import platform
import socket
import subprocess
import sys
import threading
import time

import yaml


METHODS = {
    "moham-nsga2": {
        "algorithm": "nsga2",
        "objective-mode": "legacy",
    },
    "c-nsga2": {
        "algorithm": "c-nsga2",
        "objective-mode": "slo-risk",
    },
    "c-nsga3": {
        "algorithm": "c-nsga3",
        "objective-mode": "slo-risk",
    },
    "c-taea": {
        "algorithm": "c-taea",
        "objective-mode": "slo-risk",
    },
    "scba-nsga3": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
    },
    "scba-adaptive-boundary": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "adaptive-boundary-parenting": True,
    },
    "scba-simple-candidate": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.10,
        "reference-adaptation-rate": 0.0,
    },
    "cpar-nsga3": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.10,
        "reference-adaptation-rate": 0.0,
    },
    "camo": {
        # Paper-facing name. The optimizer value remains the audited historical
        # implementation key, so this alias does not change CAMO mechanics.
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.10,
        "reference-adaptation-rate": 0.0,
    },
    "cpar-no-profile-archive": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.0,
        "reference-adaptation-rate": 0.0,
    },
    "cpar-no-repair": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.10,
        "reference-adaptation-rate": 0.0,
        "bottleneck-repair-probability": 0.0,
        "max-repair-attempts": 0,
    },
    "cpar-scalar-cv-archive": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.10,
        "boundary-archive-mode": "scalar",
        "reference-adaptation-rate": 0.0,
    },
    "cpar-random-repair": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.10,
        "reference-adaptation-rate": 0.0,
        "repair-policy": "random",
    },
    "scba-bandwidth-aware-candidate": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.10,
        "reference-adaptation-rate": 0.0,
        "bandwidth-service-aware-repair": True,
    },
    "scba-no-boundary": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.0,
    },
    "scba-boundary-005": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.05,
    },
    "scba-boundary-010": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.10,
    },
    "scba-boundary-015": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "boundary-parent-probability": 0.15,
    },
    "scba-no-adaptation": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "reference-adaptation-rate": 0.0,
    },
    "scba-no-repair": {
        "algorithm": "scba-nsga3",
        "objective-mode": "slo-risk",
        "bottleneck-repair-probability": 0.0,
        "max-repair-attempts": 0,
    },
}
PRIMARY_METHODS = [
    "moham-nsga2",
    "c-nsga2",
    "c-nsga3",
    "c-taea",
    "scba-nsga3",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root):
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest(), len(files)


def source_sha256(project_root):
    digest = hashlib.sha256()
    paths = [
        project_root / "CMakeLists.txt",
        project_root / "docker" / "Dockerfile",
        project_root / "docker" / "analysis.Dockerfile",
        project_root / "docker" / "analysis-requirements.txt",
    ]
    paths.extend(sorted((project_root / "src").glob("*.cc")))
    paths.extend(sorted((project_root / "src").glob("*.h")))
    paths.extend(sorted((project_root / "tests").glob("*.cc")))
    paths.extend(sorted((project_root / "scripts").glob("*.py")))
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest(), len(paths)


def docker_image_id(image):
    return subprocess.check_output(
        ["docker", "image", "inspect", "--format={{.Id}}", image],
        text=True,
    ).strip()


def write_manifest(path, manifest):
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command_output(command):
    """Return diagnostic command output without making capture failures fatal."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, OSError) as error:
        return {"command": command, "error": repr(error)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def read_text(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, PermissionError, OSError) as error:
        return f"UNAVAILABLE: {error!r}"


def cpufreq_snapshot():
    snapshot = {}
    cpu_root = pathlib.Path("/sys/devices/system/cpu")
    for cpu in sorted(cpu_root.glob("cpu[0-9]*"), key=lambda path: int(path.name[3:])):
        freq_root = cpu / "cpufreq"
        snapshot[cpu.name] = {
            name: read_text(freq_root / name).strip()
            for name in (
                "scaling_driver",
                "scaling_governor",
                "scaling_cur_freq",
                "scaling_min_freq",
                "scaling_max_freq",
                "cpuinfo_min_freq",
                "cpuinfo_max_freq",
            )
        }
    return snapshot


def power_supply_snapshot():
    snapshot = {}
    root = pathlib.Path("/sys/class/power_supply")
    for supply in sorted(root.glob("*")):
        snapshot[supply.name] = {
            name: read_text(supply / name).strip()
            for name in (
                "type",
                "online",
                "status",
                "capacity",
                "power_now",
                "energy_now",
                "voltage_now",
                "current_now",
            )
            if (supply / name).exists()
        }
    return snapshot


def docker_runtime_snapshot():
    listing = command_output(["docker", "ps", "-q"])
    container_ids = listing.get("stdout", "").split()
    return {
        "listing": listing,
        "containers": [
            command_output(
                [
                    "docker", "inspect", "--format",
                    "{{json .HostConfig}}", container_id,
                ]
            )
            for container_id in container_ids
        ],
    }


def lightweight_system_sample():
    return {
        "timestamp_utc": utc_now(),
        "loadavg": read_text("/proc/loadavg").strip(),
        "uptime": read_text("/proc/uptime").strip(),
        "meminfo": read_text("/proc/meminfo"),
        "vmstat": read_text("/proc/vmstat"),
        "cpu_stat": read_text("/proc/stat"),
        "cpu_frequencies": cpufreq_snapshot(),
        "power_supply": power_supply_snapshot(),
        "sensors": command_output(["sensors", "-j"]),
        "processes": command_output(
            [
                "ps", "-eo", "pid,ppid,psr,stat,pcpu,pmem,comm,args",
                "--sort=-pcpu",
            ]
        ),
        "docker_runtime": docker_runtime_snapshot(),
    }


def full_system_snapshot(project_root, binary_path):
    os_release = read_text("/etc/os-release")
    return {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "os_release": os_release,
        "kernel_cmdline": read_text("/proc/cmdline").strip(),
        "lscpu": command_output(["lscpu"]),
        "lscpu_extended": command_output(
            ["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ"]
        ),
        "logical_cpu_count": os.cpu_count(),
        "process_affinity": sorted(os.sched_getaffinity(0)),
        "meminfo": read_text("/proc/meminfo"),
        "cpu_frequencies": cpufreq_snapshot(),
        "cache_drop_interface": {
            "path": "/proc/sys/vm/drop_caches",
            "readable": os.access("/proc/sys/vm/drop_caches", os.R_OK),
            "writable": os.access("/proc/sys/vm/drop_caches", os.W_OK),
        },
        "system": lightweight_system_sample(),
        "docker_version": command_output(["docker", "version"]),
        "docker_info": command_output(["docker", "info"]),
        "git_head": command_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"]
        ),
        "git_status": command_output(
            ["git", "-C", str(project_root), "status", "--short"]
        ),
        "binary": str(binary_path),
        "binary_sha256": file_sha256(binary_path),
        "filesystem": command_output(["df", "-hT"]),
        "mounts": read_text("/proc/mounts"),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def monitor_system(stop_event, output_path, interval_seconds):
    with output_path.open("a", encoding="utf-8") as stream:
        while not stop_event.is_set():
            json.dump(lightweight_system_sample(), stream, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            stop_event.wait(interval_seconds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_config", type=pathlib.Path)
    parser.add_argument("results_root", type=pathlib.Path)
    parser.add_argument("--image", default="moham:local")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=8,
        help="OpenMP threads assigned to each concurrent MOHaM run.",
    )
    parser.add_argument(
        "--capture-output",
        action="store_true",
        help="Write each binary stdout/stderr stream to run.log.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help=(
            "Keep CSV/statistical outputs but omit large per-individual "
            "Pareto mapping files."
        ),
    )
    parser.add_argument(
        "--slo-multiplier",
        type=float,
        help="Override the static D=alpha*C_ref multiplier in generated configs.",
    )
    parser.add_argument(
        "--scheduler-generations",
        type=int,
        help="Override global-scheduler num-generations in every generated config.",
    )
    parser.add_argument(
        "--scheduler-population-size",
        type=int,
        help="Override global-scheduler population-size in every generated config.",
    )
    parser.add_argument(
        "--scheduler-immigrant-population-size",
        type=int,
        help=(
            "Override global-scheduler immigrant-population-size in every "
            "generated config."
        ),
    )
    parser.add_argument(
        "--method-scheduler-generations",
        nargs="+",
        metavar="METHOD=GENERATIONS",
        help=(
            "Override num-generations for selected methods after applying the "
            "global scheduler budget; intended for equal-wall-time sensitivity "
            "experiments."
        ),
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=PRIMARY_METHODS
    )
    parser.add_argument(
        "--shared-mapping-cache",
        type=pathlib.Path,
        help=(
            "Host directory containing medea/. It is mounted read-only and "
            "used by every method and seed."
        ),
    )
    parser.add_argument(
        "--cpu-affinity",
        help=(
            "Docker cpuset expression applied to every run, for example "
            "10,12,14. The value is recorded in the manifest."
        ),
    )
    parser.add_argument(
        "--system-monitor-interval",
        type=float,
        default=0.0,
        help=(
            "Save load, memory, CPU frequency, process and Docker snapshots "
            "at this interval in seconds. Zero disables periodic monitoring."
        ),
    )
    parser.add_argument(
        "--cache-protocol",
        choices=[
            "uncontrolled",
            "shared-mapping-cache-no-page-drop",
            "warmup-excluded-shared-mapping-cache-no-page-drop",
        ],
        default="uncontrolled",
        help="Explicit cache-state protocol recorded in the manifest.",
    )
    parser.add_argument(
        "--task-order",
        choices=["fixed", "balanced-rotation"],
        default="fixed",
        help=(
            "Execution order within each seed. Balanced rotation gives every "
            "method each ordinal position equally often when seed count permits."
        ),
    )
    parser.add_argument(
        "--wall-clock-limit-seconds",
        type=float,
        help=(
            "Graceful optimizer wall-clock budget shared by all methods. No "
            "new true evaluation starts after the deadline; in-flight "
            "evaluations drain before final outputs are written."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_base_config(config):
    workloads = config.get("workloads", [])
    constraints = config["architecture"].get("constraints", {})
    deadlines = constraints.get("workload-deadlines")
    references = constraints.get("slo-reference-latencies")
    slo_count = len(deadlines or references or [])
    if slo_count != len(workloads):
        raise ValueError(
            "The base config must provide exactly one static SLO deadline or "
            "reference latency per input DNN."
        )
    if any(seed <= 0 for seed in config.get("_requested_seeds", [])):
        raise ValueError("All experiment seeds must be positive.")


def main():
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.omp_threads <= 0:
        raise ValueError("--omp-threads must be positive.")
    if args.system_monitor_interval < 0:
        raise ValueError("--system-monitor-interval cannot be negative.")
    if (
        args.wall_clock_limit_seconds is not None
        and args.wall_clock_limit_seconds <= 0
    ):
        raise ValueError("--wall-clock-limit-seconds must be positive.")
    if args.cpu_affinity and args.jobs != 1:
        raise ValueError(
            "Controlled CPU affinity currently requires --jobs 1 so runs do "
            "not compete for the same cpuset."
        )
    budget_values = [
        args.scheduler_generations,
        args.scheduler_population_size,
        args.scheduler_immigrant_population_size,
    ]
    if any(value is not None for value in budget_values):
        if any(value is None for value in budget_values):
            raise ValueError(
                "Scheduler budget overrides must specify generations, "
                "population size, and immigrant population size together."
            )
        if any(value <= 0 for value in budget_values):
            raise ValueError("Scheduler budget overrides must be positive.")
    method_generation_overrides = {}
    for entry in args.method_scheduler_generations or []:
        try:
            method, raw_generations = entry.split("=", 1)
            generations = int(raw_generations)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "--method-scheduler-generations entries must use "
                "METHOD=GENERATIONS."
            ) from error
        if method not in args.methods:
            raise ValueError(
                "Method-specific generation override targets a method that is "
                f"not selected: {method}"
            )
        if generations <= 0:
            raise ValueError("Method-specific generations must be positive.")
        if method in method_generation_overrides:
            raise ValueError(
                f"Duplicate method-specific generation override: {method}"
            )
        method_generation_overrides[method] = generations
    if method_generation_overrides and args.scheduler_generations is None:
        raise ValueError(
            "Method-specific generation overrides require the complete global "
            "scheduler budget overrides."
        )
    project_root = pathlib.Path(__file__).resolve().parents[1]
    base_config_path = args.base_config.resolve()
    results_root = args.results_root.resolve()
    shared_mapping_cache = (
        args.shared_mapping_cache.resolve() if args.shared_mapping_cache else None
    )
    if shared_mapping_cache and not (shared_mapping_cache / "medea").is_dir():
        raise ValueError(
            f"--shared-mapping-cache must contain medea/: {shared_mapping_cache}"
        )
    with base_config_path.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    if args.slo_multiplier is not None:
        if args.slo_multiplier <= 0:
            raise ValueError("--slo-multiplier must be positive.")
        base["architecture"]["constraints"]["slo-multiplier"] = args.slo_multiplier
    base["_requested_seeds"] = args.seeds
    validate_base_config(base)
    del base["_requested_seeds"]

    scenario = base_config_path.stem
    experiment_id = hashlib.sha256(
        str(results_root).encode("utf-8")
    ).hexdigest()[:12]
    generated_root = results_root / "generated-configs" / scenario
    generated_root.mkdir(parents=True, exist_ok=True)

    source_hash, source_file_count = source_sha256(project_root)
    cache_hash = None
    cache_file_count = 0
    if shared_mapping_cache:
        cache_hash, cache_file_count = tree_sha256(shared_mapping_cache)
    binary_path = project_root / "build" / "moham"
    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "scenario": scenario,
        "experiment_id": experiment_id,
        "base_config": str(base_config_path),
        "base_config_sha256": file_sha256(base_config_path),
        "source_sha256": source_hash,
        "source_file_count": source_file_count,
        "binary": str(binary_path),
        "binary_sha256": file_sha256(binary_path),
        "docker_image": args.image,
        "docker_image_id": docker_image_id(args.image),
        "methods": args.methods,
        "seeds": args.seeds,
        "jobs": args.jobs,
        "omp_threads_per_run": args.omp_threads,
        "cpu_affinity": args.cpu_affinity,
        "omp_proc_bind": "close" if args.cpu_affinity else None,
        "omp_places": "cores" if args.cpu_affinity else None,
        "system_monitor_interval_seconds": args.system_monitor_interval,
        "cache_protocol": args.cache_protocol,
        "task_order": args.task_order,
        "wall_clock_limit_seconds": args.wall_clock_limit_seconds,
        "capture_output": args.capture_output,
        "metrics_only": args.metrics_only,
        "global_scheduler_overrides": (
            {
                "num-generations": args.scheduler_generations,
                "population-size": args.scheduler_population_size,
                "immigrant-population-size": (
                    args.scheduler_immigrant_population_size
                ),
            }
            if args.scheduler_generations is not None
            else None
        ),
        "method_scheduler_generation_overrides": method_generation_overrides,
        "shared_mapping_cache": (
            str(shared_mapping_cache) if shared_mapping_cache else None
        ),
        "shared_mapping_cache_sha256": cache_hash,
        "shared_mapping_cache_file_count": cache_file_count,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hostname": socket.gethostname(),
            "logical_cpu_count": os.cpu_count(),
            "process_affinity": sorted(os.sched_getaffinity(0)),
        },
    }

    tasks = []
    generated_config_hashes = {}
    execution_order = []
    for seed_index, seed in enumerate(args.seeds):
        cache_container = (
            "/shared-mapping-cache/medea"
            if shared_mapping_cache
            else f"/results/mapping-cache/{scenario}/seed-{seed}/medea"
        )
        methods_for_seed = list(args.methods)
        if args.task_order == "balanced-rotation" and methods_for_seed:
            shift = seed_index % len(methods_for_seed)
            methods_for_seed = methods_for_seed[shift:] + methods_for_seed[:shift]
        execution_order.append({"seed": seed, "methods": methods_for_seed})
        for method in methods_for_seed:
            method_spec = METHODS[method]
            run_config = copy.deepcopy(base)
            run_config["search"]["seed"] = seed
            run_config["search"]["mapping-cache-dir"] = cache_container
            run_config["search"]["trust-mapping-cache"] = bool(shared_mapping_cache)
            scheduler = run_config["search"]["global-scheduler"]
            if args.scheduler_generations is not None:
                scheduler.update(
                    {
                        "num-generations": method_generation_overrides.get(
                            method, args.scheduler_generations
                        ),
                        "population-size": args.scheduler_population_size,
                        "immigrant-population-size": (
                            args.scheduler_immigrant_population_size
                        ),
                    }
                )
            scheduler.update(method_spec)
            if args.wall_clock_limit_seconds is not None:
                scheduler["wall-clock-limit-seconds"] = (
                    args.wall_clock_limit_seconds
                )
            if args.metrics_only:
                scheduler["output-pareto-files"] = False

            config_host = generated_root / f"{method}-seed-{seed}.yaml"
            with config_host.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(run_config, stream, sort_keys=False)
            generated_config_hashes[config_host.name] = file_sha256(config_host)

            output_relative = pathlib.Path("runs") / scenario / method / f"seed-{seed}"
            output_host = results_root / output_relative
            stats_file = output_host / "moham" / "moham.stats.txt"
            if stats_file.exists():
                print(f"SKIP completed: {method} seed={seed}")
                continue

            config_container = (
                pathlib.Path("/results")
                / "generated-configs"
                / scenario
                / config_host.name
            )
            output_container = pathlib.Path("/results") / output_relative
            command = [
                "docker",
                "run",
                "--rm",
                "--label",
                f"moham.experiment_id={experiment_id}",
            ]
            if args.cpu_affinity:
                command.extend(["--cpuset-cpus", args.cpu_affinity])
            command.extend(
                [
                "-e",
                f"OMP_NUM_THREADS={args.omp_threads}",
                "-e",
                f"OMP_PROC_BIND={'close' if args.cpu_affinity else 'false'}",
                "-e",
                f"OMP_PLACES={'cores' if args.cpu_affinity else 'threads'}",
                "-v",
                f"{project_root}:/workspaces/moham",
                "-v",
                f"{results_root}:/results",
                ]
            )
            if shared_mapping_cache:
                command.extend(
                    ["-v", f"{shared_mapping_cache}:/shared-mapping-cache:ro"]
                )
            command.extend(
                [
                    "-w",
                    "/workspaces/moham/experiments",
                    args.image,
                    "../build/moham",
                    "-c",
                    str(config_container),
                    "-o",
                    str(output_container),
                ]
            )
            tasks.append((command, method, seed, output_host))

    manifest["execution_order"] = execution_order
    manifest["generated_config_sha256"] = generated_config_hashes
    manifest_path = results_root / "experiment_manifest.yaml"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            previous = yaml.safe_load(stream)
        identity_keys = [
            "scenario",
            "base_config_sha256",
            "source_sha256",
            "binary_sha256",
            "docker_image_id",
            "methods",
            "seeds",
            "jobs",
            "omp_threads_per_run",
            "cpu_affinity",
            "omp_proc_bind",
            "omp_places",
            "system_monitor_interval_seconds",
            "cache_protocol",
            "task_order",
            "execution_order",
            "wall_clock_limit_seconds",
            "capture_output",
            "metrics_only",
            "global_scheduler_overrides",
            "method_scheduler_generation_overrides",
            "shared_mapping_cache_sha256",
            "generated_config_sha256",
        ]
        mismatches = [
            key for key in identity_keys
            if previous.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Refusing to mix incompatible experiment versions in "
                f"{results_root}; manifest fields differ: {mismatches}"
            )
        manifest["created_at_utc"] = previous.get(
            "created_at_utc", manifest["created_at_utc"]
        )
    write_manifest(manifest_path, manifest)

    environment_root = results_root / "environment"
    if not args.dry_run:
        environment_root.mkdir(parents=True, exist_ok=True)
        write_json(
            environment_root / "system_snapshot_start.json",
            full_system_snapshot(project_root, binary_path),
        )
        write_json(
            environment_root / "protocol.json",
            {
                "schema_version": 1,
                "created_at_utc": utc_now(),
                "purpose": "controlled runtime and environment evidence rerun",
                "cpu_affinity": args.cpu_affinity,
                "jobs": args.jobs,
                "omp_threads_per_run": args.omp_threads,
                "omp_proc_bind": "close" if args.cpu_affinity else None,
                "omp_places": "cores" if args.cpu_affinity else None,
                "task_order": args.task_order,
                "wall_clock_limit_seconds": args.wall_clock_limit_seconds,
                "cache_protocol": args.cache_protocol,
                "cache_notes": (
                    "The shared mapping cache is mounted read-only. CPU caches "
                    "cannot be preserved between processes. Linux page cache is "
                    "not dropped because that requires privileged global state "
                    "changes. A separate excluded sentinel run establishes the "
                    "declared warm-up state when requested."
                ),
                "background_monitor_interval_seconds": (
                    args.system_monitor_interval
                ),
            },
        )

    monitor_stop = threading.Event()
    monitor_thread = None
    if not args.dry_run and args.system_monitor_interval > 0:
        monitor_thread = threading.Thread(
            target=monitor_system,
            args=(
                monitor_stop,
                environment_root / "system_monitor.jsonl",
                args.system_monitor_interval,
            ),
            daemon=True,
        )
        monitor_thread.start()

    def run_task(task):
        command, method, seed, output_host = task
        print(f"RUN {method} seed={seed}", flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            return
        output_host.mkdir(parents=True, exist_ok=True)
        write_json(
            output_host / "environment_before.json",
            lightweight_system_sample(),
        )
        with (output_host / "command.txt").open(
            "w", encoding="utf-8"
        ) as command_file:
            command_file.write(" ".join(command) + "\n")
        started_at = utc_now()
        started_monotonic = time.monotonic()
        returncode = None
        try:
            if args.capture_output:
                log_path = output_host / "run.log"
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        check=False,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            else:
                completed = subprocess.run(command, check=False)
            returncode = completed.returncode
        finally:
            finished_at = utc_now()
            duration_seconds = time.monotonic() - started_monotonic
            write_json(
                output_host / "run_metadata.json",
                {
                    "schema_version": 1,
                    "method": method,
                    "seed": seed,
                    "started_at_utc": started_at,
                    "finished_at_utc": finished_at,
                    "wall_clock_seconds_external": duration_seconds,
                    "returncode": returncode,
                    "command": command,
                    "command_sha256": hashlib.sha256(
                        "\0".join(command).encode("utf-8")
                    ).hexdigest(),
                },
            )
            write_json(
                output_host / "environment_after.json",
                lightweight_system_sample(),
            )
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        print(f"DONE {method} seed={seed}", flush=True)

    try:
        if args.jobs == 1:
            for task in tasks:
                run_task(task)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=args.jobs
            )
            interrupted = False
            try:
                futures = [executor.submit(run_task, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            except KeyboardInterrupt:
                interrupted = True
                completed = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-q",
                        "--filter",
                        f"label=moham.experiment_id={experiment_id}",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                containers = completed.stdout.split()
                if containers:
                    subprocess.run(
                        ["docker", "stop"] + containers,
                        check=False,
                    )
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=not interrupted, cancel_futures=True)
    finally:
        if monitor_thread is not None:
            monitor_stop.set()
            monitor_thread.join(timeout=max(5.0, args.system_monitor_interval + 1.0))
        if not args.dry_run:
            write_json(
                environment_root / "system_snapshot_end.json",
                full_system_snapshot(project_root, binary_path),
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
