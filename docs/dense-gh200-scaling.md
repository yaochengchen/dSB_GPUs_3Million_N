# Dense GH200 capacity-scaling protocol

## Purpose

This experiment is a capacity and throughput study, not a solution-quality
benchmark.  It asks how far a single GH200 process can advance an explicit,
dense FP32 dSB system when the matrix crosses the local-HBM boundary.

## Instance

For every `N`, the code materialises a symmetric matrix with zero diagonal and
deterministic high-entropy weights

```
J_ij = J_ji in [-1/sqrt(N), +1/sqrt(N)], seed 42.
```

The matrix is approximately 100% nonzero. CSR would require a column index in
addition to every value and is therefore deliberately not used. FastHare is
also disabled. Do not describe the data as information-theoretically
incompressible; describe it as dense high-entropy data with negligible benefit
from sparse storage.

## Default sweep

```
N = 10k, 20k, 40k, 80k, 120k, 160k, 200k, 240k, 300k
batch = 1, 8, 64          (BATCH_LIST)
precision = FP32, FP16    (PRECISION_LIST)
steps = 50
warm-up steps = 1
repeats = 3
seed = 42
physical GPU = 1
```

`batch=1` is kept because at `N=300k` one matrix is 360 GB (FP32) and a
single dSB step must stream it; it is the capacity point.  It is no longer
the only point: at batch 1 every formulation is a bandwidth-bound stream of
J and the PyTorch GEMV already sits at the HBM roofline (4.1 TB/s measured),
so nothing can beat it there and the comparison says nothing about the
solver.  The batch sweep shows the per-replica step cost falling with batch
until the GEMM turns compute bound, which is the quantity that matters for
dSB.  FP16 storage halves the bytes per step and moves the HBM boundary from
about N=176k to about N=250k on a 144 GB part.

## Compared paths

- public PyTorch dense baseline using the ordinary CUDA/HBM allocator, run
  at the same dtype as each of our rows;
- dense `gemm` (cuBLAS; TF32 for FP32 storage, HMMA for FP16), HBM only --
  the only dense formulation that parallelises over rows, and therefore the
  only one that can fill the GPU at small batch;
- dense `cluster`, HBM only, kept as the shared-memory rung of the ladder
  (it launches `batch x cluster` blocks and is expected to lose at batch=1);
- proposed `auto`, which now resolves to `gemm` below 64 replicas and to
  block/cluster above, and chooses HBM/Grace placement.

The first run of this experiment (batch=1, FP32, no `gemm`) measured
`cluster` at 229 GB/s effective against 4122 GB/s for the GEMV: 8 blocks on
132 SMs.  That result is explained, not disputed, and is why `gemm` is in
the list.

All failures are data: `oom`, `launch-limit`, `grace-unavailable`, or
`timeout`. Never replace a failed row with an extrapolated time.

## Defensible claim

Use this wording if the measured output supports it:

> Among the evaluated implementations, only the proposed adaptive GH200 path
> completed the largest explicit dense instances.

Do not claim that every possible baseline is fundamentally unable to run. A
separately engineered Grace-aware out-of-core baseline could also be written.

## Beyond HBM (v9 runs)

The sweep above stops where the matrix leaves HBM (FP32: 160k, FP16: 240k).
Two further placements of `gemm` continue past that point, selected with
`MATRIX_MEMORY` in `scripts/run_dense_scaling.sh` (see `run.md` §5 for the
exact commands and `results/README.md` for the directories):

- `auto`: the whole matrix in pinned Grace memory on the GPU's NUMA node,
  streamed every step through a 2 GiB HBM staging buffer -- 371 GB/s at every
  n and both precisions, FP32 to 300k (360 GB) and FP16 to 450k (405 GB);
- `hybrid`: HBM filled first (127 GB), only the remaining rows in Grace
  memory -- 1.4--3.7x faster than `auto`, `T = M_H/4.2 TB/s + M_G/371 GB/s`
  within 4%.

The dense ±1 bit format takes the same hybrid placement to 3,000,000
variables on two GPUs (`docs/bit-dense-multi.md`). The PyTorch baseline uses
the ordinary HBM allocator and stops at the HBM limit; the defensible claim
stays as worded above.
