#!/usr/bin/env python3
"""CPU-only checks for publication-comparison matching and speedups."""

from __future__ import annotations

import csv
import io
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from summarize_fasthare_ablation import FIELDS, emit, lookup_public  # noqa: E402


class FastHareAblationTest(unittest.TestCase):
    def test_match_and_speedup(self) -> None:
        common = {
            "instance": "synthetic", "agents": "200", "steps": "800",
            "precision": "fp32", "time_field": "total_s",
            "variant": "discrete", "storage": "dense",
            "best_objective": "10", "tts99_s": "20",
        }
        baseline = dict(common, implementation="public", reduction="1",
                        median_time_s="4")
        candidate = dict(common, implementation="dsb-gpu", reduction="1",
                         variant="gemm", median_time_s="1", tts99_s="5",
                         best_objective="11", median_preprocess_s="0.1",
                         median_reconstruction_s="0.01", reduction_ratio="0.5",
                         n_before_reduction="10", n_reduced="5")
        self.assertIs(lookup_public([baseline, candidate], candidate, "1"), baseline)

        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        emit(writer, "solver-after-fasthare", baseline, candidate)
        row = next(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(row["median_time_speedup"], "4")
        self.assertEqual(row["tts99_speedup"], "4")
        self.assertEqual(row["objective_delta"], "1")


if __name__ == "__main__":
    unittest.main()
