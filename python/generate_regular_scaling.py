#!/usr/bin/env python3
"""Random d-regular Max-Cut instances (non-trivial optimum), same file format as
generate_bipartite_scaling.py: "N E" header, then "u v 1" per edge, 1-based.
Requires networkx (pip install networkx)."""
import argparse, pathlib
import networkx as nx

def generate(path: pathlib.Path, n: int, degree: int, seed: int) -> None:
    g = nx.random_regular_graph(degree, n, seed=seed)   # simple graph, no loops
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1 << 20) as f:
        f.write(f"{n} {g.number_of_edges()}\n")
        for u, v in g.edges():
            a, b = (u, v) if u < v else (v, u)
            f.write(f"{a + 1} {b + 1} 1\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=pathlib.Path)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--degree", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.output.exists() and not a.force:
        print(f"exists, keeping: {a.output}")
    else:
        generate(a.output, a.n, a.degree, a.seed); print(a.output)
