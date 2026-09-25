#!/usr/bin/env python3
"""Summarize batched-run CSV and calculate TTS99 when a target is supplied."""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument(
        "--time-field",
        choices=("total_s", "solver_s", "wall_s"),
        default="total_s",
    )
    args = parser.parse_args()

    groups = defaultdict(list)
    with open(args.csv_file, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row.get(name, "") for name in (
                "implementation", "instance", "agents", "steps", "precision", "variant",
                "storage", "reduction"
            ))
            groups[key].append(row)

    fields = [
        "implementation", "instance", "agents", "steps", "precision", "variant",
        "storage", "reduction",
        "runs", "best_objective", "median_objective", "successes", "p_batch",
        "median_time_s", "tts99_s", "time_field", "n_original", "edges", "target",
        "median_solver_s", "median_time_per_step_s", "median_edge_updates_per_s",
        "median_preprocess_s", "median_reconstruction_s", "median_wall_s",
        "median_setup_s", "median_evaluation_s",
        "reduction_ratio", "n_before_reduction", "n_reduced",
        "gpu_memory_bytes",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for key, rows in sorted(groups.items()):
        objectives = [float(row["objective"]) for row in rows]
        times = [float(row[args.time_field]) for row in rows]
        success_values = [row["success"] for row in rows if row["success"] != ""]
        successes = sum(int(value) for value in success_values) if success_values else None
        p_batch = successes / len(success_values) if success_values else None
        median_time = statistics.median(times)
        if p_batch is None or p_batch <= 0.0:
            tts99 = "" if p_batch is None else "inf"
        elif p_batch >= 1.0:
            tts99 = "%.9g" % median_time
        else:
            tts99 = "%.9g" % (median_time * math.log(0.01) / math.log(1.0 - p_batch))
        optional_median = lambda name: (
            "%.9g" % statistics.median(float(row[name]) for row in rows if row.get(name, ""))
            if any(row.get(name, "") for row in rows) else ""
        )
        writer.writerow(dict(zip(fields[:8], key), **{
            "runs": len(rows),
            "best_objective": "%.17g" % max(objectives),
            "median_objective": "%.17g" % statistics.median(objectives),
            "successes": "" if successes is None else successes,
            "p_batch": "" if p_batch is None else "%.9g" % p_batch,
            "median_time_s": "%.9g" % median_time,
            "tts99_s": tts99,
            "time_field": args.time_field,
            "n_original": rows[0].get("n_original", ""),
            "edges": rows[0].get("edges", ""),
            "target": rows[0].get("target", ""),
            "median_solver_s": optional_median("solver_s"),
            "median_time_per_step_s": optional_median("time_per_step_s"),
            "median_edge_updates_per_s": optional_median("edge_updates_per_s"),
            "median_preprocess_s": optional_median("preprocess_s"),
            "median_reconstruction_s": optional_median("reconstruction_s"),
            "median_wall_s": optional_median("wall_s"),
            "median_setup_s": optional_median("setup_s"),
            "median_evaluation_s": optional_median("evaluation_s"),
            "reduction_ratio": rows[0].get("reduction_ratio", ""),
            "n_before_reduction": rows[0].get("n_before_reduction", ""),
            "n_reduced": rows[0].get("n_reduced", ""),
            "gpu_memory_bytes": rows[0].get("gpu_memory_bytes", ""),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
