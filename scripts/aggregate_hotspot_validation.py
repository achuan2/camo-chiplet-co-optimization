#!/usr/bin/env python3
"""Aggregate and de-duplicate per-method HotSpot validation reports."""

import argparse
import csv
import json
import math
import pathlib
import statistics

from validate_hotspot import average_ranks, pearson


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_root", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.validation_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    raw_count = 0
    for manifest_path in sorted(
        root.glob("*/hotspot_validation_manifest.json")
    ):
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        method = manifest_path.parent.name
        candidates = {
            int(candidate["individual"]): candidate
            for candidate in manifest["candidate_manifests"]
        }
        with pathlib.Path(manifest["results_csv"]).open(
            newline="", encoding="utf-8"
        ) as stream:
            for row in csv.DictReader(stream):
                raw_count += 1
                candidate = candidates[int(row["individual"])]
                key = (
                    candidate["files"]["floorplan_sha256"],
                    candidate["files"]["power_trace_sha256"],
                )
                if key in seen:
                    continue
                seen.add(key)
                compact = float(row["compact_peak_c"])
                hotspot = float(row["hotspot_peak_c"])
                records.append(
                    {
                        "method": method,
                        "individual": int(row["individual"]),
                        "compact_peak_c": compact,
                        "hotspot_peak_c": hotspot,
                        "signed_error_c": compact - hotspot,
                        "absolute_error_c": abs(compact - hotspot),
                        "chiplets": int(row["accelerators"]),
                        "floorplan_sha256": key[0],
                        "power_trace_sha256": key[1],
                    }
                )
    if not records:
        raise ValueError(f"No HotSpot validation reports below {root}")

    signed = [row["signed_error_c"] for row in records]
    absolute = [row["absolute_error_c"] for row in records]
    compact = [row["compact_peak_c"] for row in records]
    hotspot = [row["hotspot_peak_c"] for row in records]
    summary = {
        "raw_case_count": raw_count,
        "unique_case_count": len(records),
        "mae_c": statistics.mean(absolute),
        "rmse_c": math.sqrt(statistics.mean(error**2 for error in signed)),
        "max_absolute_error_c": max(absolute),
        "max_underprediction_c": max(0.0, max(-error for error in signed)),
        "mean_signed_error_c": statistics.mean(signed),
        "pearson": pearson(compact, hotspot),
        "spearman": pearson(
            average_ranks(compact), average_ranks(hotspot)
        ),
    }
    output_csv = output_dir / "hotspot_validation_aggregate.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    output_json = output_dir / "hotspot_validation_aggregate.json"
    with output_json.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "validation_root": str(root),
                "summary": summary,
                "cases_csv": str(output_csv),
            },
            stream,
            indent=2,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
