#!/usr/bin/env python3
"""Run the recorded scientific analysis on the portable selected-run raw CSV tree."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(str(source), str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/ctaea-sensitivity")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sensitivity-reanalysis")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    raw = args.raw.resolve()
    index = json.loads((raw / "index.json").read_text())
    if len(index["runs"]) != 660:
        raise SystemExit("Expected the complete 660-run selected input")
    destination = args.output.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit("Choose a new, empty --output directory")
    destination.mkdir(parents=True, exist_ok=True)
    work = destination / "work"
    work.mkdir()
    script = ROOT / "analysis/reference/analyze_supplement.py"
    shutil.copy2(str(script), str(work / "analyze_supplement.py"))
    shutil.copy2(str(ROOT / "analysis/reference/experiment_plan.yaml"), str(work / "experiment_plan.yaml"))
    utility = work / "source/scripts/analyze_experiment_matrix.py"
    utility.parent.mkdir(parents=True)
    shutil.copy2(str(ROOT / "scripts/analyze_experiment_matrix.py"), str(utility))
    completed = {}
    for item in index["runs"]:
        directory = Path(item["directory"])
        if directory.is_absolute() or ".." in directory.parts:
            raise SystemExit("Unsafe run directory in index")
        output = work / "inputs" / directory
        for filename, identity in item["files"].items():
            source = raw / directory / filename
            if hashlib.sha256(source.read_bytes()).hexdigest() != identity["sha256"]:
                raise SystemExit("Raw input checksum mismatch: " + str(source))
            link_or_copy(source, output / "moham" / filename)
        completed.setdefault(item["setting"], []).append({
            "task": {"method": item["method"], "seed": item["seed"]},
            "output": str(output), "external_time_s": item["external_time_s"],
        })
    for setting, rows in completed.items():
        path = work / "formal" / setting / "completed_runs.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(rows, indent=2))
    if args.prepare_only:
        print("PASS: 660 raw inputs prepared with verified identities; no statistical analysis executed.")
        return
    # The saved script ends with private handoff-document/ZIP generation.
    # Execute its unchanged scientific-analysis prefix, omitting that bundler.
    code = script.read_text(encoding="utf-8")
    marker = "summary=['# CAMO supplementary experiment handoff'"
    if code.count(marker) != 1:
        raise SystemExit("Recorded analysis-script boundary changed")
    scientific = code.split(marker, 1)[0]
    namespace = {"__name__": "__main__", "__file__": str(work / "analyze_supplement.py")}
    exec(compile(scientific, str(work / "analyze_supplement.py"), "exec"), namespace)
    print("Analysis complete: " + str(work / "analysis"))


if __name__ == "__main__":
    main()
