#!/usr/bin/env python3
"""Check file hashes, frozen source/workload payloads, and result-family keys."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-manifest", action="store_true")
    args = parser.parse_args()
    errors, checked = [], 0
    if not args.skip_manifest:
        manifest = ROOT / "MANIFEST.sha256"
        if not manifest.exists():
            errors.append("MANIFEST.sha256 is missing")
        else:
            for line in manifest.read_text(encoding="utf-8").splitlines():
                expected, name = line.split("  ", 1)
                path = ROOT / name
                if not path.is_file() or digest(path) != expected:
                    errors.append("File hash mismatch: " + name)
                checked += 1
    assets = json.loads((ROOT / "provenance/source-assets.json").read_text())["files"]
    source_count = 0
    for item in assets:
        if "/.git/" in item["path"] or item["path"].endswith("/.git"):
            continue
        path = ROOT / item["path"]
        if not path.is_file() or digest(path) != item["sha256"]:
            errors.append("Frozen source mismatch: " + item["path"])
        source_count += 1
    catalog = json.loads((ROOT / "provenance/families.json").read_text())
    total = 0
    for family in catalog:
        with (ROOT / family["directory"] / "run_metrics.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != family["runs"]:
            errors.append("Run count: " + family["id"])
        keys = [(row["scenario"], row["method"], row["seed"]) for row in rows]
        if len(keys) != len(set(keys)):
            errors.append("Duplicate run key: " + family["id"])
        if rows and "raw_output" in rows[0]:
            errors.append("Machine-local output path in public metrics: " + family["id"])
        total += len(rows)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        "PASS: %d file hashes; %d frozen source/workload payloads; "
        "%d separate result families; %d run-metric records."
        % (checked, source_count, len(catalog), total)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
