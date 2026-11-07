#!/usr/bin/env python3
"""Regenerate descriptive summaries and Figure 2 source values from saved metrics.

Recorded, family-specific p-values remain in each paired_tests.csv.
"""
import argparse
import collections
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    "hypervolume_5d", "igd_plus_5d", "feasible_evaluations_total",
    "unique_feasible_designs_total", "final_common_nd_feasible_objective_count",
]


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def quantile(values, q):
    values = sorted(v for v in values if not math.isnan(v))
    if not values:
        return math.nan
    position = (len(values) - 1) * q
    low, high = int(math.floor(position)), int(math.ceil(position))
    if low == high or values[low] == values[high]:
        return values[low]
    if math.isinf(values[high]):
        return values[high]
    return values[low] + (position - low) * (values[high] - values[low])


def write(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def camo(name):
    return name.lower() in {"camo", "cpar-nsga3", "scba-nsga3"}


def label(name):
    if camo(name):
        return "CAMO"
    return {
        "cpar-nsga3": "CAMO", "scba-nsga3": "CAMO",
        "c-nsga2": "C-NSGA-II", "c-nsga3": "C-NSGA-III",
        "c-taea": "C-TAEA", "moham-nsga2": "MOHaM", "nsga2": "MOHaM",
    }.get(name, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduced-results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, families, checks, errors = [], {}, 0, []
    for family in json.loads((ROOT / "provenance/families.json").read_text()):
        directory = ROOT / family["directory"]
        groups = collections.defaultdict(list)
        for row in read(directory / "run_metrics.csv"):
            groups[(row["scenario"], row["method"])].append(row)
        saved = {
            (row["scenario"], row["method"], row["metric"]): row
            for row in read(directory / "summary_metrics.csv")
        }
        current = {}
        for (scenario, method), rows in groups.items():
            for metric in METRICS + ["evaluations"]:
                if metric not in rows[0]:
                    continue
                values = [float(row[metric]) for row in rows if row[metric] != ""]
                median, q1, q3 = [quantile(values, q) for q in (0.5, 0.25, 0.75)]
                summaries.append({
                    "family": family["id"], "scenario": scenario, "method": method,
                    "metric": metric, "n": len(values), "median": median,
                    "q1": q1, "q3": q3, "iqr": q3 - q1,
                })
                current[(method, metric)] = median
                expected = saved.get((scenario, method, metric))
                if expected:
                    for key, actual in [("median", median), ("q1", q1), ("q3", q3)]:
                        value = float(expected[key])
                        if math.isfinite(actual) and math.isfinite(value):
                            checks += 1
                            if not math.isclose(actual, value, rel_tol=2e-12, abs_tol=2e-12):
                                errors.append("/".join([family["id"], method, metric, key]))
        families[family["id"]] = current
    figure = []
    for family, scenario in [
        ("primary-normal", "Normal"), ("primary-temperature", "Temperature-tight"),
        ("primary-power", "Power-tight"), ("primary-slo", "SLO-tight"),
    ]:
        values = families[family]
        for metric in ["hypervolume_5d", "igd_plus_5d"]:
            target = next(v for (m, k), v in values.items() if k == metric and camo(m))
            candidates = [(m, v) for (m, k), v in values.items() if k == metric and not camo(m)]
            best = max(candidates, key=lambda x: x[1]) if metric == "hypervolume_5d" else min(candidates, key=lambda x: x[1])
            baseline, value = best
            percent = 100 * ((target - value) / value if metric == "hypervolume_5d" else (value - target) / value)
            figure.append({
                "panel": "a", "scenario": scenario, "method": "CAMO", "metric": metric,
                "value": percent, "unit": "percent", "baseline": label(baseline),
                "not_significant": scenario == "SLO-tight" and metric == "igd_plus_5d",
            })
    for (method, metric), value in families["wallclock"].items():
        if metric in ("hypervolume_5d", "igd_plus_5d", "evaluations"):
            figure.append({
                "panel": "b", "scenario": "Normal (900 s)", "method": label(method),
                "metric": metric, "value": value,
                "unit": "count" if metric == "evaluations" else "normalized quality",
                "baseline": "", "not_significant": False,
            })
    write(args.output / "summary_metrics.csv", summaries)
    write(args.output / "figure2_data.csv", figure)
    report = {
        "finite_summary_values_checked": checks, "mismatches": errors,
        "summaries_regenerated": len(summaries),
        "p_values": "Recorded family-specific paired_tests.csv; no pooling or silent recomputation.",
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2))
    if errors:
        print("Summary mismatch: " + ", ".join(errors), file=sys.stderr)
        return 1
    print("PASS: %d descriptive rows; %d finite stored values matched. Outputs: %s" % (len(summaries), checks, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
