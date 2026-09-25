#!/usr/bin/env python3
"""Benchmark the public dSB package on a Stanford G-set instance.

The output columns match ``solve_gset --csv`` so the two implementations can
be concatenated and passed directly to ``summarize_tts.py``.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, TextIO

import numpy as np

from fasthare_io import FastHareReduction, lift_spins, load_reduction
import sb_schedule


CSV_FIELDS = [
    "implementation", "instance", "repeat", "seed", "n_original",
    "n_before_reduction", "n_reduced", "edges", "agents", "steps",
    "precision", "variant",
    "storage", "reduction", "early_stopping", "objective", "target",
    "success", "preprocess_s", "setup_s", "solver_s", "reconstruction_s",
    "evaluation_s", "total_s", "wall_s", "reduction_ratio",
    "time_per_step_s", "edge_updates_per_s", "nnz_solver",
    "coupling_bytes", "gpu_memory_bytes",
]


@dataclass
class GsetInstance:
    name: str
    coupling: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_weight: np.ndarray


def load_gset(path: pathlib.Path) -> GsetInstance:
    """Read and validate the standard ``n m`` followed by ``u v w`` format."""
    tokens = path.read_text(encoding="utf-8").split()
    if len(tokens) < 2:
        raise ValueError("invalid G-set header: %s" % path)

    n = int(tokens[0])
    m = int(tokens[1])
    if n <= 0 or m < 0:
        raise ValueError("invalid G-set dimensions: %s" % path)
    if len(tokens) != 2 + 3 * m:
        raise ValueError(
            "expected %d edge fields, found %d in %s"
            % (3 * m, len(tokens) - 2, path)
        )

    edge_u = np.empty(m, dtype=np.int64)
    edge_v = np.empty(m, dtype=np.int64)
    edge_weight = np.empty(m, dtype=np.float64)
    coupling = np.zeros((n, n), dtype=np.float32)

    for edge_index in range(m):
        offset = 2 + 3 * edge_index
        u = int(tokens[offset]) - 1
        v = int(tokens[offset + 1]) - 1
        weight = float(tokens[offset + 2])
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("vertex index out of range at edge %d" % (edge_index + 1))
        if u == v:
            raise ValueError("self-loop at edge %d" % (edge_index + 1))
        if not np.isfinite(weight):
            raise ValueError("non-finite weight at edge %d" % (edge_index + 1))
        edge_u[edge_index] = u
        edge_v[edge_index] = v
        edge_weight[edge_index] = weight
        coupling[u, v] += weight
        coupling[v, u] += weight

    scale = float(np.max(np.abs(coupling))) if coupling.size else 0.0
    if scale == 0.0:
        scale = 1.0
    coupling /= scale
    return GsetInstance(path.stem, coupling, edge_u, edge_v, edge_weight)


def synchronize(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_once(
    torch, Ising, instance: GsetInstance, args, repeat: int,
    reduction: Optional[FastHareReduction]
):
    seed = args.seed + repeat
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    dtype = {"float16": torch.float16, "float32": torch.float32,
             "float64": torch.float64}[args.dtype]
    synchronize(torch, args.device)
    total_start = time.perf_counter()

    preprocess_s = 0.0 if reduction is None else reduction.preprocess_s
    fully_reduced = reduction is not None and reduction.fully_reduced
    setup_start = time.perf_counter()
    model = None
    if not fully_reduced:
        solver_coupling = (
            -instance.coupling if reduction is None else -reduction.coupling
        )
        model = Ising(
            solver_coupling,
            np.zeros(solver_coupling.shape[0], dtype=np.float32),
            dtype=dtype,
            device=args.device,
        )
        synchronize(torch, args.device)
    setup_s = time.perf_counter() - setup_start

    if fully_reduced:
        reduced_spins = torch.empty(
            (0, args.agents), dtype=dtype, device=args.device
        )
        solver_s = 0.0
    else:
        solve_start = time.perf_counter()
        reduced_spins = model.minimize(
            agents=args.agents,
            max_steps=args.steps,
            mode="discrete",
            heated=False,
            verbose=False,
            early_stopping=args.early_stopping,
        )
        synchronize(torch, args.device)
        solver_s = time.perf_counter() - solve_start

    reconstruction_start = time.perf_counter()
    spins = (
        reduced_spins
        if reduction is None
        else lift_spins(torch, reduced_spins, reduction, args.device)
    )
    synchronize(torch, args.device)
    reconstruction_s = time.perf_counter() - reconstruction_start

    evaluation_start = time.perf_counter()
    edge_u = torch.as_tensor(instance.edge_u, dtype=torch.long, device=args.device)
    edge_v = torch.as_tensor(instance.edge_v, dtype=torch.long, device=args.device)
    weights = torch.as_tensor(instance.edge_weight, dtype=dtype, device=args.device)
    cut_values = 0.5 * torch.sum(
        weights[:, None] * (1.0 - spins[edge_u, :] * spins[edge_v, :]),
        dim=0,
    )
    best = float(torch.max(cut_values).item())
    synchronize(torch, args.device)
    evaluation_s = time.perf_counter() - evaluation_start
    measured_wall_s = time.perf_counter() - total_start
    total_s = preprocess_s + solver_s + reconstruction_s
    wall_s = preprocess_s + measured_wall_s

    target = args.target
    return {
        "implementation": "simulated-bifurcation-2.0.0",
        "instance": instance.name,
        "repeat": repeat,
        "seed": seed,
        "n_original": instance.coupling.shape[0],
        "n_before_reduction": instance.coupling.shape[0],
        "n_reduced": (
            instance.coupling.shape[0] if reduction is None else reduction.n_reduced
        ),
        "edges": instance.edge_weight.size,
        "agents": args.agents,
        "steps": args.steps,
        "precision": {"float16": "fp16", "float32": "fp32",
                      "float64": "fp64"}[args.dtype],
        "variant": args.sb_variant,
        "storage": "dense",
        "reduction": int(reduction is not None),
        "early_stopping": int(args.early_stopping),
        "objective": "%.17g" % best,
        "target": "" if target is None else "%.17g" % target,
        "success": "" if target is None else int(best >= target - args.atol),
        "preprocess_s": "%.9g" % preprocess_s,
        "setup_s": "%.9g" % setup_s,
        "solver_s": "%.9g" % solver_s,
        "reconstruction_s": "%.9g" % reconstruction_s,
        "evaluation_s": "%.9g" % evaluation_s,
        "total_s": "%.9g" % total_s,
        "wall_s": "%.9g" % wall_s,
        "reduction_ratio": "%.9g" % (
            0.0 if reduction is None else reduction.reduction_ratio
        ),
        "time_per_step_s": "%.9g" % (solver_s / args.steps),
        "edge_updates_per_s": "",
        "nnz_solver": int(np.count_nonzero(
            instance.coupling if reduction is None else reduction.coupling
        )),
        "coupling_bytes": (
            instance.coupling.nbytes
            if reduction is None else reduction.coupling.nbytes
        ),
        "gpu_memory_bytes": "",
    }


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=pathlib.Path)
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"),
                        default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--target", type=float)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--reduction-file", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    sb_schedule.add_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.agents <= 0 or args.steps <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError(
            "agents, steps, and repeats must be positive; warmup must be nonnegative"
        )

    try:
        import torch
        import simulated_bifurcation as sb
        from simulated_bifurcation.core import Ising
    except ImportError as error:
        print(
            "missing baseline dependency; install a CUDA PyTorch build, then "
            "`python -m pip install -r requirements-baseline.txt`",
            file=sys.stderr,
        )
        raise SystemExit(2) from error

    # Global package state: set once, before warm-up, for every repeat.
    args.sb_variant = sb_schedule.configure(sb, args)
    print("simulated-bifurcation schedule=%s env=%s"
          % (args.sb_schedule, sb_schedule.describe(sb)), file=sys.stderr)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if getattr(sb, "__version__", None) != "2.0.0":
        print(
            "warning: benchmark was written for simulated-bifurcation 2.0.0",
            file=sys.stderr,
        )

    instance = load_gset(args.instance)
    reduction = (
        None
        if args.reduction_file is None
        else load_reduction(args.reduction_file, "gset")
    )
    if reduction is not None:
        if reduction.name != instance.name:
            raise ValueError("reduction instance name does not match input")
        if reduction.n_original != instance.coupling.shape[0]:
            raise ValueError("reduction original size does not match input")
    for warmup_index in range(args.warmup):
        run_once(
            torch, Ising, instance, args, -1 - warmup_index, reduction
        )

    stream: TextIO
    close_stream = False
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("w", newline="", encoding="utf-8")
        close_stream = True
    else:
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for repeat in range(args.repeats):
            writer.writerow(
                run_once(torch, Ising, instance, args, repeat, reduction)
            )
    finally:
        if close_stream:
            stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
