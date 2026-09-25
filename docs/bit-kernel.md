# The bit-fused kernel, and where each variant wins

This note records what the K2000 and dense-scaling measurements showed, what
was changed in response, and what to expect from the next run.

## What the measurements said

**K2000 (N=2000, B=200, 800 steps, fp32).**  The `block` kernel took
353.6 us/step.  That number is `B * N^2 * 4 bytes / (L2 read bandwidth)` =
3.2 GB / 9 TB/s = 355 us to within 0.5%: the kernel is bound by L2 bandwidth
at an arithmetic intensity of exactly 1, because each block owns one replica
and each J element it loads feeds one FMA.  `gemm` (cuBLAS, FP32 storage,
CUDA cores) took 72 us; the FFMA roofline for the same product is 24 us.

**Dense scaling (N=10k..160k, B=1).**  `cluster` was 18-56x slower than the
PyTorch GEMV because a batch of 1 launches `cluster` (<= 8) blocks on 132
SMs.  229 GB/s effective is 8/132 of the 4.1 TB/s the GEMV reached.  The
`gemm` variant was not in the run.

Neither result is a tuning problem.  The per-replica shared-memory design
has no operating point at which it beats a tensor-core GEMM on real-valued
J: small N loses on arithmetic, large N loses on J re-reads, small B loses on
occupancy.

**First measurement of v3 (GH200, `bench 2000 200 100 pm1`, us/step):**
global-sync 3514 | gemm fp16 **10** | gemm fp32 71 | gemm tf32 20 |
bit (v3) 16 | block 330 | cluster 1266.  Two things this settled:

* The honest baseline is cuBLAS FP16 at 10 us -- faster than the 15-25
  guessed earlier.
* The v3 bit kernel at 16 us lost to it.  Its budget was 201,600 (XOR,POPC)
  per block per step with two scalar shared loads each: POPC at 16/SM/clk is
  6.4 us and the loads were another ~6.4 us that did not overlap.  The load
  half is fixed in v4 (below); the POPC half is a floor.  On Hopper, XOR+POPC
  on CUDA cores is about the same rate as HMMA -- the exact path that beats
  FP16 outright is INT8 IMMA, added in v4 as `--precision=int8`.

Also found by `selftest`: the second-run mismatch of exactly 2.0 on the
sparse ternary case was the sign(0) convention.  A clipped state restarted at
pump 0 lands on x == 0.0 exactly whenever the integer `acc` is 0; the FP32
kernels said sign(0) = 0, the 1-bit path cannot.  The code base now uses
sign(x) = +1 if x > 0 else -1 everywhere (see `dsb/common.hpp`).  No normal
run can produce x == 0 (it needs p = 0 and a clipped state), so earlier
results are unaffected.

## What was changed

### `int8` gemm (new in v4) -- `src/gemm.cu`, `Precision::INT8`

J in int8 ({-1,0,+1}), S in int8 (+-1), int32 accumulation: exact, and on
the IMMA pipe at ~2x the FP16 rate.  `cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N,
..., CUBLAS_COMPUTE_32I)` needs S replica-major ([B][ldk]) and m = batch a
multiple of 4; the sign/update kernels transpose through a 32x33 shared tile
so every global access stays coalesced.  The trajectory is bit-identical to
the FP32 kernels (selftest asserts 0).  Gemm only; the solver refuses other
variants at int8.  If cuBLAS returns status 15 (NOT_SUPPORTED) on your
build, the fallback is cublasLtMatmul with explicit row-major layouts -- say
so and I will write it.

### `bit` variant (new in v3, rewritten in v4) -- `src/bitfused.cu`

For J in {-1, 0, +1} -- every G-set instance and K2000 -- the coupling term
is a product of two +-1 vectors and has an exact integer identity:

    sum_k a_k b_k  =  n - 2 * popcount(a XOR b)

so one row of the coupling term is 63 XOR+POPC pairs at N=2000 instead of
2000 FFMAs, and the answer is exact (integer sums are exact in FP32 too, so
the trajectories are bit-identical to the FP32 kernels; `selftest` asserts a
difference of exactly 0).

Layout: a block owns R = 16 rows (or 32 when the 16-row grid cannot be
co-resident) x all replicas; ceil(N/R) blocks -- 125 at N=2000.  J's
bitplane slice (4 KB at N=2000) and the block's x, y stay in shared memory
for the whole run, so J is read from device memory once per solve.  The sign
bitmap is [W][B] words, published as one uint16/uint32 per (block, replica)
with no conflicts, double buffered in device memory, one `grid.sync()` per
step.  Sparse ternary instances use two planes (P = +1 edges, M = -1 edges)
and `2*(popc(P&S) - popc(M&S)) - (|P|-|M|)`.

v4 layout: the shared bitmap is replica-major with a padded row stride
Wp = 4*odd words (68 at N=2000), so the inner loop reads the J row (a warp
broadcast) and the sign row 16 bytes at a time with no bank conflicts; the
device bitmap is replica-major too, which makes the 16-row half-word layout
a plain aligned 32-bit read.  Loads drop 4x; POPC stays at 16*200*68 =
217,600 per block per step -> 6.9 us floor with 125 blocks.  Expect ~8-9 us
measured, i.e. roughly level with cuBLAS FP16, exact, with 500 KB of J.  The
bit kernel's cost is N*B*W popcounts regardless of sparsity, so `auto` in
`solve_gset` uses it only at density >= 4%.

Fit: `W*B*4 + R*B*8 + planes*R*W*4 <= 227 KB` per block and ceil(N/R)
blocks co-resident.  At B=200 that covers N up to ~4000 -- all of G1-G54 and
K2000; `make_bit_plan` picks R=16 when it fits and R=32 otherwise.

### Tensor cores in `gemm`

`Options::tf32` (default on) sets `CUBLAS_COMPUTE_32F_FAST_TF32` for FP32
storage.  FP16 storage already used HMMA but no K2000/G-set run had used it.
`gemm:fp16` is now in the default variant list of the runners as the honest
library baseline; `selftest` compares the TF32 path on energy only.

### `csr-row`: one barrier per step

The persistent CSR kernel snapshotted signs in a separate pass and paid two
`grid.sync()` per step; it now writes the next sign as part of the update
into a second buffer.  On large sparse G-set (N >= 7000) the barriers, not
the arithmetic, are the step time.

### `auto` dispatch

* `Solver` (dense): `bit` when J is ternary and the plan fits; otherwise
  `gemm`.  Block/cluster are never chosen automatically any more: on GH200
  there is no (N, batch) at which they beat a tensor-core GEMM on
  real-valued J.  They remain available explicitly for the ladder.
* `solve_gset`: `bit` when ternary, fits, and density >= 4%; else `csr-row`
  when density < 25%; else `gemm`.  Decided before any N x N is built.
* CSV rows from `--variant=auto` are labelled `auto->bit` etc.

### Dense scaling

* `--precision=fp16` in `dense_scaling` and the PyTorch baseline: half the
  bytes per step; the HBM boundary moves from N ~ 176k to ~ 250k.
* `scripts/run_dense_scaling.sh` sweeps `BATCH_LIST` (1 8 64) and
  `PRECISION_LIST` (fp32 fp16), includes `gemm`, and reaches N = 300k so the
  Grace spill actually happens.

## What to run, in order

    make clean && make -j
    ./selftest 512 4 50            # bit and int8 vs block must show 0.000e+00
    ./selftest 2048 8 50           # W=64, two blocks per SM
    ./bench 2000 200 100 pm1       # the K2000 ladder in one table (now with gemm/int8)

    STEPS_LIST="800" REPEATS=20 ./run_K2000.sh
    STEPS_LIST="800" REPEATS=20 ./run_Gset.sh 1 22 43 48 55 60 70 81

    N_LIST="20000 80000 160000 200000 240000" BATCH_LIST="1 8 64" \
      PRECISION_LIST="fp32 fp16" CPP_MODES="gemm auto" \
      scripts/run_dense_scaling.sh

## K2000, us/step: measured (v3) and expected (v4)

| path                        | measured v3 | expected v4 | note                        |
|-----------------------------|------------:|------------:|-----------------------------|
| block fp32                  |         330 |         330 | L2-bound, intensity 1       |
| PyTorch SB 2.0.0            |       169.5 |       169.5 | (from the 800-step run)     |
| gemm fp32, CUDA cores       |          71 |          71 | `--no-tf32`                 |
| gemm fp32 + TF32            |          20 |          20 |                             |
| gemm fp16 (HMMA)            |      **10** |          10 | the baseline to quote       |
| bit                         |          16 |        ~8-9 | exact; 500 KB J on chip     |
| **gemm int8 (IMMA)**        |           - |      **~5** | exact; the proposed fast path |

Then the ladder for the paper is PyTorch 169 -> cuBLAS FP16 10 (17x, with
rounding) -> int8 ~5 (~30x, exact) with `bit` as the exact minimum-memory
variant.  If int8 does not land near 5 us: `ncu --metrics
sm__pipe_tensor_op_imma_cycles_active` tells whether the IMMA kernel was
picked; if `cublasGemmEx` returned NOT_SUPPORTED, see the int8 section.

## Scope, stated plainly

* Exact and fast: J in {0, +-1} (all of G-set, K2000, unweighted Max-Cut):
  `gemm:int8` (fastest) and `bit` (smallest).
* Small integer weights |J| <= 127: int8 is still exact (int32 accumulation
  over N <= 16M terms cannot overflow); `bit` would need k planes.
* Real-valued J (QPLIB, the dense-scaling matrices): `gemm` with fp16/TF32.
  Neither exact path applies and `Solver` refuses them.
