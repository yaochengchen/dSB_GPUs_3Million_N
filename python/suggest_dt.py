#!/usr/bin/env python3
"""Suggest a dSB step size dt for one instance (G-set/edge list or QPLIB).

Why dt has to depend on the instance
------------------------------------
dsb-gpu integrates, per step,

    y += (-(1 - p) x + xi * J sign(x)) dt ;  x += dt y      (xi = 0.5 sqrt(N-1)/|J|_F)

Along an eigenvector of J with eigenvalue lambda < 0 the coupling term is a
restoring force of stiffness about  k = xi * |lambda_min(J)|.  A symplectic-
Euler oscillator of stiffness k goes unstable once dt^2 * k passes a small
constant; with the sign() force the measured edge is dt^2 * k ~ 2.7.  Past it
the uniform mode flips every step, hits the walls and all replicas end with
every spin on one side: cut ~ 0.  That is the G1 result of the v4 run
(objective 29 against 11624 in every variant): G1 has mean degree 48, so for
J = -W the Perron mode gives k ~ 3.5 and dt = 1 is past the edge
(the proxy sweep collapses between dt = 0.85 and 0.9).

Typical values of k (same xi formula in dsb-gpu and in SB 2.0.0):

    4-regular +-1 (G11-G13, G32-G34, G48-G50, reduced QPLIB)   ~0.9
    K2000 (complete +-1), dense uniform real-valued J             ~1.0
    sparse random, mean degree ~5 (G55-G66)                        ~1.1
    random, mean degree ~20 (G22-G31, G43-G47)                     ~2.3
    random, mean degree ~48 (G1-G10)                               ~3.5
    d-regular bipartite, d = 100 (run_large_scale.sh)              ~5.0

The suggestion targets dt^2 * k = 1.2 (the best-quality region in the proxy
sweep, well inside the stable side), capped to [0.25, 1.25]:

    dt = clip(sqrt(1.2 / k), 0.25, 1.25)

It is a starting point, not an optimum; the sweep showed quality within
~0.1 % over a band of +-0.25 around it.

Usage
-----
    python3 python/suggest_dt.py data/Gset/data/G1            # -> 0.55
    python3 python/suggest_dt.py --format=qplib QPLIB_3506.qplib
    python3 python/suggest_dt.py --verbose G22
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

TARGET = 1.2        # dt^2 * k aimed for
DT_MIN, DT_MAX = 0.25, 1.25


def gset_coupling(path: pathlib.Path) -> sp.csr_matrix:
    """J = -W from an 'n m / u v w' edge list, sparse (no N x N allocation)."""
    with path.open("r", encoding="utf-8") as stream:
        n, m = (int(t) for t in stream.readline().split()[:2])
        data = np.loadtxt(stream, dtype=np.float64, ndmin=2, max_rows=m)
    if data.shape[0] != m:
        raise ValueError("expected %d edges, found %d" % (m, data.shape[0]))
    u = data[:, 0].astype(np.int64) - 1
    v = data[:, 1].astype(np.int64) - 1
    w = data[:, 2]
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    vals = -np.concatenate([w, w])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def qplib_coupling(path: pathlib.Path) -> sp.csr_matrix:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from benchmark_public_dsb import load_qplib  # same transform as solve_qplib
    return sp.csr_matrix(load_qplib(path).solver_coupling.astype(np.float64))


def stiffness(j: sp.csr_matrix) -> tuple[float, float, float]:
    n = j.shape[0]
    fro = math.sqrt(float(j.multiply(j).sum()))
    if n < 2 or fro == 0.0:
        return 0.0, 0.0, 0.0
    xi = 0.5 * math.sqrt(n - 1) / fro
    if n <= 64:
        lam = float(np.linalg.eigvalsh(j.toarray())[0])
    else:
        lam = float(spla.eigsh(j, k=1, which="SA", tol=1e-3,
                               return_eigenvectors=False)[0])
    return xi, lam, xi * max(0.0, -lam)


def suggest(k: float) -> float:
    if k <= 0.0:
        return DT_MAX
    return min(DT_MAX, max(DT_MIN, math.sqrt(TARGET / k)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("instance", type=pathlib.Path)
    parser.add_argument("--format", choices=("auto", "gset", "qplib"), default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    fmt = args.format
    if fmt == "auto":
        fmt = "qplib" if args.instance.suffix == ".qplib" else "gset"
    j = qplib_coupling(args.instance) if fmt == "qplib" else gset_coupling(args.instance)
    xi, lam, k = stiffness(j)
    dt = suggest(k)
    if args.verbose:
        print("n=%d nnz=%d xi=%.6g lambda_min(J)=%.6g k=xi*|lambda_min|=%.4g "
              "edge dt~%.3g suggested dt=%.2f"
              % (j.shape[0], j.nnz, xi, lam, k,
                 math.sqrt(2.7 / k) if k > 0 else float("inf"), dt),
              file=sys.stderr)
    print("%.2f" % dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
