#!/usr/bin/env python3
"""Benchmark the public third-party dSB implementation on a QPLIB instance.

This uses bqth29/simulated-bifurcation-algorithm (PyPI package
``simulated-bifurcation``), not unreleased Toshiba source code.  It consumes
the same QPLIB-to-Ising transformation as ``solve_qplib`` and emits the same
CSV columns so whole batched runs can be compared and summarized as TTS.
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
    "n_before_reduction", "n_reduced", "agents", "steps", "precision",
    "variant", "storage", "reduction", "early_stopping", "objective",
    "target", "success", "preprocess_s", "setup_s", "solver_s",
    "reconstruction_s", "evaluation_s", "total_s", "wall_s",
    "reduction_ratio", "gpu_memory_bytes",
]


@dataclass
class QplibInstance:
    name: str
    coupling: np.ndarray
    field: np.ndarray
    offset: float
    solver_coupling: np.ndarray


def load_qplib(path: pathlib.Path) -> QplibInstance:
    """Mirror src/qplib.cpp's supported QPLIB subset exactly."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError("file too short: %s" % path)
    n_edges = int(lines[4].split()[0])
    if n_edges < 0:
        raise ValueError("negative edge count")

    edges = []
    for line in lines[5 : 5 + n_edges]:
        a, b, w = line.split()[:3]
        edges.append((int(a), int(b), float(w)))
    if not edges:
        raise ValueError("no vertices parsed")

    default_field = float(lines[5 + n_edges].split()[0])
    n_fields = int(lines[6 + n_edges].split()[0])
    field_entries = {}
    for line in lines[7 + n_edges : 7 + n_edges + n_fields]:
        index, value = line.split()[:2]
        field_entries[int(index) - 1] = float(value)

    n = max(max(a, b) for a, b, _ in edges)
    j = np.zeros((n, n), dtype=np.float64)
    for a, b, weight in edges:
        a -= 1
        b -= 1
        w = weight / 2.0
        j[a, b] = w
        j[b, a] = w

    h = np.zeros(n, dtype=np.float64)
    if field_entries:
        h.fill(default_field)
        for index, value in field_entries.items():
            if 0 <= index < n:
                h[index] = value

    j *= 1.0 / 8.0
    offset = -float(j.sum()) - float(h.sum()) / 2.0
    h = h * 0.5 + 2.0 * j.sum(axis=1)
    j *= -2.0
    h *= -1.0
    scale = max(float(np.max(np.abs(j))), float(np.max(np.abs(h))))
    if scale == 0.0:
        scale = 1.0
    solver_coupling = np.zeros((n + 1, n + 1), dtype=np.float32)
    solver_coupling[:n, :n] = (-j / scale).astype(np.float32)
    solver_coupling[:n, n] = (-h / scale).astype(np.float32)
    solver_coupling[n, :n] = (-h / scale).astype(np.float32)
    return QplibInstance(path.stem, j, h, offset, solver_coupling)


def synchronize(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_once(
    torch, Ising, instance: QplibInstance, args, repeat: int,
    reduction: Optional[FastHareReduction]
):
    seed = args.seed + repeat
    torch.manual_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    synchronize(torch, args.device)
    total_start = time.perf_counter()

    preprocess_s = 0.0 if reduction is None else reduction.preprocess_s
    fully_reduced = reduction is not None and reduction.fully_reduced
    setup_start = time.perf_counter()
    model = None
    if not fully_reduced:
        solver_coupling = (
            instance.solver_coupling
            if reduction is None
            else -reduction.coupling
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
        synchronize(torch, args.device)
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
    if reduction is None:
        gauge = reduced_spins[instance.field.size, :]
        spins = reduced_spins[: instance.field.size, :] * gauge[None, :]
    else:
        spins = lift_spins(torch, reduced_spins, reduction, args.device)
    synchronize(torch, args.device)
    reconstruction_s = time.perf_counter() - reconstruction_start

    evaluation_start = time.perf_counter()
    coupling = torch.as_tensor(
        instance.coupling, dtype=dtype, device=args.device
    )
    field = torch.as_tensor(instance.field, dtype=dtype, device=args.device)
    values = (
        -0.5 * torch.sum((coupling @ spins) * spins, dim=0)
        - field @ spins
        - instance.offset
    )
    best = float(torch.max(values).item())
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
        "n_original": instance.field.size,
        "n_before_reduction": instance.field.size + 1,
        "n_reduced": (
            instance.field.size + 1 if reduction is None else reduction.n_reduced
        ),
        "agents": args.agents,
        "steps": args.steps,
        "precision": "fp32" if args.dtype == "float32" else "fp64",
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
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
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
        raise ValueError("agents, steps, and repeats must be positive; warmup must be nonnegative")

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
        print("warning: benchmark was written for simulated-bifurcation 2.0.0", file=sys.stderr)

    instance = load_qplib(args.instance)
    reduction = (
        None
        if args.reduction_file is None
        else load_reduction(args.reduction_file, "qplib")
    )
    if reduction is not None:
        if reduction.name != instance.name:
            raise ValueError("reduction instance name does not match input")
        if reduction.n_original != instance.field.size:
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
