#!/usr/bin/env python3
"""CPU-only checks for the FastHare interchange format."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from fasthare_io import load_reduction  # noqa: E402


class FastHareIoTest(unittest.TestCase):
    def test_load_qplib_reduction(self) -> None:
        reduction = load_reduction(
            ROOT / "tests" / "fixtures" / "fasthare_qplib.txt", "qplib"
        )
        self.assertEqual(reduction.name, "synthetic")
        self.assertEqual((reduction.n_original, reduction.n_standard), (2, 3))
        self.assertEqual(reduction.n_reduced, 2)
        self.assertAlmostEqual(reduction.alpha, 0.2)
        self.assertAlmostEqual(reduction.preprocess_s, 0.125)
        self.assertAlmostEqual(reduction.reduction_ratio, 1.0 / 3.0)
        np.testing.assert_array_equal(reduction.sign, [1, -1, 1])
        np.testing.assert_array_equal(reduction.spin_map, [0, 1, 0])
        np.testing.assert_array_equal(
            reduction.coupling, np.asarray([[0.0, 2.5], [2.5, 0.0]])
        )


if __name__ == "__main__":
    unittest.main()
