#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--plot")
    args = parser.parse_args()

    with open(args.raw, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    groups: dict[tuple[str, str, str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        requested = row.get("requested_variant") or "unknown"
        selected = row.get("selected_variant") or requested
        label = f"auto->{selected}" if requested == "auto" else selected
        memory = row.get("selected_memory") or row.get("requested_memory") or "unknown"
        precision = row.get("precision") or "fp32"
        batch = int(row.get("batch") or 1)
        groups[(row["implementation"], label, memory, precision, batch,
                int(row["n"]))].append(row)

    fields = [
        "implementation", "variant", "memory", "precision", "batch", "n",
        "status", "runs",
        "median_gpu_s", "median_time_per_step_s",
        "median_dense_interactions_per_s", "median_effective_matrix_GB_s",
        "matrix_bytes", "hbm_bytes", "grace_bytes", "error",
    ]
    summary = []
    for key in sorted(groups, key=lambda item: (item[3], item[4], item[5], item[0], item[1], item[2])):
        implementation, variant, memory, precision, batch, n = key
        members = groups[key]
        completed = [row for row in members if row.get("status") == "completed"]

        def median(field: str) -> str:
            values = [float(row[field]) for row in completed if row.get(field)]
            return f"{statistics.median(values):.9g}" if values else ""

        status = "completed" if completed else members[0].get("status", "error")
        summary.append({
            "implementation": implementation,
            "variant": variant,
            "memory": memory,
            "precision": precision,
            "batch": batch,
            "n": n,
            "status": status,
            "runs": len(completed),
            "median_gpu_s": median("gpu_s"),
            "median_time_per_step_s": median("time_per_step_s"),
            "median_dense_interactions_per_s": median("dense_interactions_per_s"),
            "median_effective_matrix_GB_s": median("effective_matrix_GB_s"),
            "matrix_bytes": members[0].get("matrix_bytes", ""),
            "hbm_bytes": members[0].get("hbm_bytes", ""),
            "grace_bytes": members[0].get("grace_bytes", ""),
            "error": "" if completed else members[0].get("error", ""),
        })

    with open(args.summary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return 0
        series: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in summary:
            if row["status"] == "completed" and row["median_time_per_step_s"]:
                label = (f'{row["implementation"]}:{row["variant"]}:{row["memory"]}'
                         f':{row["precision"]}:b{row["batch"]}')
                series[label].append((int(row["n"]), float(row["median_time_per_step_s"])))
        figure, axis = plt.subplots(figsize=(8.0, 5.2))
        for label, points in sorted(series.items()):
            points.sort()
            axis.plot([p[0] for p in points], [p[1] for p in points],
                      marker="o", label=label)
        axis.set_xlabel("Dense problem size N")
        axis.set_ylabel("Median GPU time per dSB step (s)")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(args.plot, dpi=180)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
