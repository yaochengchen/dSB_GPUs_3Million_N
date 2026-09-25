#!/usr/bin/env python3
"""Generate deterministic simple d-regular bipartite Max-Cut instances."""

import argparse
import pathlib
import random


def generate(path: pathlib.Path, n: int, degree: int, seed: int) -> None:
    if n <= 0 or n % 2:
        raise ValueError("N must be a positive even integer")
    half = n // 2
    if degree <= 0 or degree > half:
        raise ValueError("degree must be in [1, N/2]")
    rng = random.Random(seed)
    left = list(range(1, half + 1))
    right = list(range(half + 1, n + 1))
    rng.shuffle(left)
    rng.shuffle(right)
    shifts = rng.sample(range(half), degree)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
        stream.write(f"{n} {n * degree // 2}\n")
        for i, u in enumerate(left):
            for shift in shifts:
                stream.write(f"{u} {right[(i + shift) % half]} 1\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--degree", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        print(f"exists, keeping: {args.output}")
        return 0
    generate(args.output, args.n, args.degree, args.seed)
    print(f"generated: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
