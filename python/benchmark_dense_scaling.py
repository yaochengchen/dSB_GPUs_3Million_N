#!/usr/bin/env python3
"""PyTorch dense-HBM baseline for the GH200 capacity experiment.

This intentionally uses the normal PyTorch CUDA allocator.  It records OOM as
data instead of aborting the suite.  The random values need not be bitwise
identical to the C++ matrix because this experiment measures capacity and step
throughput, not solution quality; shape, density, precision, batch and steps
are identical.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time

import torch


FIELDS = [
    "implementation", "n", "repeat", "seed", "batch", "steps",
    "precision", "requested_variant", "selected_variant",
    "requested_memory", "selected_memory", "cluster", "matrix_bytes",
    "hbm_bytes", "grace_bytes", "hbm_free_before", "hbm_total",
    "generation_s", "gpu_s", "time_per_step_s",
    "dense_interactions_per_s", "effective_matrix_GB_s", "status", "error",
]


def emit(writer: csv.DictWriter, **values: object) -> None:
    row = {field: "" for field in FIELDS}
    row.update(values)
    writer.writerow(row)
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                        help="storage dtype of J and the state (matches the "
                             "C++ --precision flag)")
    args = parser.parse_args()

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    element = 2 if args.precision == "fp16" else 4

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    matrix_bytes = args.n * args.n * element
    free_before = total = 0

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        free_before, total = torch.cuda.mem_get_info(device)

        start_generation = time.perf_counter()
        coupling = torch.empty((args.n, args.n), dtype=dtype, device=device)
        coupling.uniform_(-1.0 / math.sqrt(args.n),
                          1.0 / math.sqrt(args.n))
        generation_s = time.perf_counter() - start_generation
        x = torch.empty((args.n, args.batch), dtype=dtype,
                        device=device).uniform_(-0.01, 0.01)
        y = torch.empty_like(x).uniform_(-0.01, 0.01)
        xi = math.sqrt(3.0) / 2.0

        @torch.no_grad()
        def run(count: int) -> float:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for step in range(count):
                pump = 0.0 if args.steps == 1 else step / (args.steps - 1)
                acc = coupling @ torch.sign(x)
                y.add_((-(1.0 - pump) * x + xi * acc))
                x.add_(y)
                clipped = x.abs() > 1.0
                x[clipped] = x[clipped].sign()
                y[clipped] = 0.0
            end.record()
            end.synchronize()
            return begin.elapsed_time(end) * 1.0e-3

        if args.warmup_steps:
            run(args.warmup_steps)
        for repeat in range(args.repeats):
            gpu_s = run(args.steps)
            interactions = args.n * (args.n - 1) * args.batch * args.steps
            streamed = matrix_bytes * args.batch * args.steps
            emit(
                writer,
                implementation="python-pytorch-dense",
                n=args.n,
                repeat=repeat,
                seed=args.seed,
                batch=args.batch,
                steps=args.steps,
                precision=args.precision,
                requested_variant="python",
                selected_variant="python",
                requested_memory="hbm",
                selected_memory="hbm",
                cluster=0,
                matrix_bytes=matrix_bytes,
                hbm_bytes=matrix_bytes + 2 * args.n * args.batch * element,
                grace_bytes=0,
                hbm_free_before=free_before,
                hbm_total=total,
                generation_s=generation_s,
                gpu_s=gpu_s,
                time_per_step_s=gpu_s / args.steps,
                dense_interactions_per_s=interactions / gpu_s,
                effective_matrix_GB_s=streamed / gpu_s / 1.0e9,
                status="completed",
            )
        return 0
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        text = str(error).replace("\n", " ").replace(",", ";")
        status = "oom" if "out of memory" in text.lower() else "error"
        emit(
            writer,
            implementation="python-pytorch-dense",
            n=args.n,
            repeat=0,
            seed=args.seed,
            batch=args.batch,
            steps=args.steps,
            precision=args.precision,
            requested_variant="python",
            selected_variant="python",
            requested_memory="hbm",
            matrix_bytes=matrix_bytes,
            hbm_free_before=free_before,
            hbm_total=total,
            status=status,
            error=text,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
