#!/usr/bin/env python3
"""Build the three publication comparisons from a TTS summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Iterable, Optional


FIELDS = [
    "comparison", "instance", "agents", "steps", "precision", "time_field",
    "baseline_implementation", "baseline_variant", "baseline_storage",
    "baseline_reduction", "candidate_implementation", "candidate_variant",
    "candidate_storage", "candidate_reduction", "baseline_median_time_s",
    "candidate_median_time_s", "median_time_speedup", "baseline_tts99_s",
    "candidate_tts99_s", "tts99_speedup", "baseline_best_objective",
    "candidate_best_objective", "objective_delta", "preprocess_s",
    "reconstruction_s", "reduction_ratio", "n_before_reduction", "n_reduced",
]


def number(row: dict[str, str], name: str) -> Optional[float]:
    value = row.get(name, "")
    if value in ("", "inf", "-inf", "nan"):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def ratio(numerator: Optional[float], denominator: Optional[float]) -> str:
    if numerator is None or denominator is None or denominator <= 0.0:
        return ""
    return "%.9g" % (numerator / denominator)


def lookup_public(
    rows: Iterable[dict[str, str]], candidate: dict[str, str], reduction: str
) -> Optional[dict[str, str]]:
    for row in rows:
        if row.get("implementation") == "dsb-gpu":
            continue
        if row.get("reduction") != reduction:
            continue
        # There is no public int8 dSB; an int8 candidate (exact integer
        # arithmetic) is compared against the public fp32 row.
        wanted_precision = candidate.get("precision", "")
        if wanted_precision == "int8":
            wanted_precision = "fp32"
        if row.get("precision", "") != wanted_precision:
            continue
        if all(row.get(field, "") == candidate.get(field, "") for field in (
            "instance", "agents", "steps", "time_field"
        )):
            return row
    return None


def emit(
    writer: csv.DictWriter, label: str, baseline: dict[str, str],
    candidate: dict[str, str]
) -> None:
    baseline_time = number(baseline, "median_time_s")
    candidate_time = number(candidate, "median_time_s")
    baseline_tts = number(baseline, "tts99_s")
    candidate_tts = number(candidate, "tts99_s")
    baseline_objective = number(baseline, "best_objective")
    candidate_objective = number(candidate, "best_objective")
    objective_delta = (
        "" if baseline_objective is None or candidate_objective is None
        else "%.17g" % (candidate_objective - baseline_objective)
    )
    writer.writerow({
        "comparison": label,
        "instance": candidate.get("instance", ""),
        "agents": candidate.get("agents", ""),
        "steps": candidate.get("steps", ""),
        "precision": candidate.get("precision", ""),
        "time_field": candidate.get("time_field", ""),
        "baseline_implementation": baseline.get("implementation", ""),
        "baseline_variant": baseline.get("variant", ""),
        "baseline_storage": baseline.get("storage", ""),
        "baseline_reduction": baseline.get("reduction", ""),
        "candidate_implementation": candidate.get("implementation", ""),
        "candidate_variant": candidate.get("variant", ""),
        "candidate_storage": candidate.get("storage", ""),
        "candidate_reduction": candidate.get("reduction", ""),
        "baseline_median_time_s": baseline.get("median_time_s", ""),
        "candidate_median_time_s": candidate.get("median_time_s", ""),
        "median_time_speedup": ratio(baseline_time, candidate_time),
        "baseline_tts99_s": baseline.get("tts99_s", ""),
        "candidate_tts99_s": candidate.get("tts99_s", ""),
        "tts99_speedup": ratio(baseline_tts, candidate_tts),
        "baseline_best_objective": baseline.get("best_objective", ""),
        "candidate_best_objective": candidate.get("best_objective", ""),
        "objective_delta": objective_delta,
        "preprocess_s": candidate.get("median_preprocess_s", ""),
        "reconstruction_s": candidate.get("median_reconstruction_s", ""),
        "reduction_ratio": candidate.get("reduction_ratio", ""),
        "n_before_reduction": candidate.get("n_before_reduction", ""),
        "n_reduced": candidate.get("n_reduced", ""),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv")
    args = parser.parse_args()
    with open(args.summary_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS)
    writer.writeheader()
    for candidate in rows:
        if candidate.get("implementation") != "dsb-gpu":
            continue
        reduction = candidate.get("reduction", "")
        same_mode = lookup_public(rows, candidate, reduction)
        if same_mode is not None:
            emit(
                writer,
                "solver-no-reduction" if reduction == "0"
                else "solver-after-fasthare",
                same_mode,
                candidate,
            )
        if reduction == "1":
            raw_public = lookup_public(rows, candidate, "0")
            if raw_public is not None:
                emit(writer, "end-to-end-fasthare-pipeline", raw_public, candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
