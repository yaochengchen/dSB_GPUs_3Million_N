#!/usr/bin/env python3
"""suggest_dt: stiffness k = xi * |lambda_min(J)| on graphs with known spectra."""

from __future__ import annotations

import math
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    import suggest_dt
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def write_gset(path: pathlib.Path, n: int, edges) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("%d %d\n" % (n, len(edges)))
        for u, v, w in edges:
            f.write("%d %d %d\n" % (u + 1, v + 1, w))


@unittest.skipUnless(HAVE_SCIPY, "needs scipy")
class SuggestDtTest(unittest.TestCase):
    def test_complete_graph(self) -> None:
        # Unweighted K_n, J = -W: lambda_min = -(n-1), |J|_F = sqrt(n(n-1)),
        # so k = 0.5 (n-1) / sqrt(n).  n = 100 -> k = 4.95 -> dt = 0.49.
        n = 100
        edges = [(i, j, 1) for i in range(n) for j in range(i + 1, n)]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "K100"
            write_gset(path, n, edges)
            _, lam, k = suggest_dt.stiffness(suggest_dt.gset_coupling(path))
        self.assertAlmostEqual(lam, -(n - 1), places=2)
        self.assertAlmostEqual(k, 0.5 * (n - 1) / math.sqrt(n), places=3)
        self.assertAlmostEqual(suggest_dt.suggest(k), math.sqrt(1.2 / k), places=6)

    def test_clipped(self) -> None:
        self.assertEqual(suggest_dt.suggest(0.0), suggest_dt.DT_MAX)
        self.assertEqual(suggest_dt.suggest(1e-3), suggest_dt.DT_MAX)
        self.assertEqual(suggest_dt.suggest(1e3), suggest_dt.DT_MIN)

    def test_fixture(self) -> None:
        j = suggest_dt.gset_coupling(ROOT / "tests" / "fixtures" / "gset_triangle.txt")
        dt = suggest_dt.suggest(suggest_dt.stiffness(j)[2])
        self.assertTrue(suggest_dt.DT_MIN <= dt <= suggest_dt.DT_MAX)


if __name__ == "__main__":
    unittest.main()
