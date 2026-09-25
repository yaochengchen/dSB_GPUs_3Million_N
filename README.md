# dsb-gpu

GPU implementation of **discrete Simulated Bifurcation** (dSB) for dense and
CSR Ising problems, with a GH200 HBM--Grace-memory capacity path and a dense
±1 bit-packed path that splits the coupling matrix over two GH200s (up to
3,000,000 variables). FastHare Hamiltonian reduction is kept as an interface
but is not used in any reported result.

Target: **NVIDIA GH200 / H100, `sm_90`; one GPU for everything except
`bit_dense_multi`, which uses one or two.** Nothing here is written to be
portable to older architectures.

How to run every experiment of the paper, and where its results are, is in
[`run.md`](run.md) (Chinese) and [`results/README.md`](results/README.md);
the analysis and figures are produced by the sibling `dsb_analysis/` tree.

> Dynamics: H. Goto et al., *High-performance combinatorial optimization based
> on classical mechanics*, Sci. Adv. **7**, eabe7953 (2021).
> Reduction: Nguyen et al., *FastHare*, IEEE QCE 2023.

Paper: Y. Chen, *GPU discrete simulated bifurcation on a Grace Hopper
Superchip: execution-path selection, exact low-precision arithmetic, and
scaling* (manuscript, 2026). If you use this code, please cite it; a
`CITATION.cff` will be added when the paper is published.

---

## Implementations included in the paper comparison

The comparison scripts expose the public baseline plus the existing GPU
variants. FastHare is an orthogonal on/off experiment dimension; it is not a
new solver variant.

1. public Python/PyTorch baseline (fp32 and, alongside `gemm:fp16`, fp16);
2. `gemm:int8` -- **the proposed fast path for J in {-1,0,+1}** (all of
   G-set, K2000): int8 IMMA with int32 accumulation, exact, ~2x the FP16
   rate.  See `docs/bit-kernel.md`;
3. `bit` -- the proposed minimum-memory exact path: bitplane J (N^2/8 bytes)
   resident in shared memory for the whole run, coupling term by XOR+POPC;
4. `csr-row` (persistent, one barrier per step) -- the large-sparse path;
5. `gemm` (dense cuBLAS on tensor cores: TF32 for fp32 storage, HMMA for
   fp16) -- the honest library baseline, run as `gemm:fp16` in the scripts;
6. `block`, `cluster` (dense, per-replica shared memory), `csr-block`,
   `csr-cluster` -- the ladder; kept for the ablation, not proposed;
7. `bit_dense_multi` -- the dense ±1 form of the bit path with the packed
   matrix in device memory instead of shared memory, one replica, rows split
   over one or two GH200s with each GPU's HBM filled first and the rest in its
   own Grace memory; exact, deterministic across GPU counts and placements,
   with a planted Mattis instance as an at-scale correctness check.  See
   `docs/bit-dense-multi.md`.

`auto` dispatches: `bit` when the instance is ternary and the plan fits (and,
in `solve_gset`, density >= 4%); `csr-row` when sparse; `gemm` otherwise.
`global-sync` remains an optional diagnostic control because it is much slower.

FastHare is preprocessing, CSR/dense is storage, and block/cluster is GPU
scheduling.  They are recorded in separate CSV columns rather than treated as
the same concept.

## Dense kernels, same arithmetic

Every variant computes the same thing each step. They differ in where the state
lives and how the per-step barrier is done — and, as a consequence, in how many
times each byte of `J` gets dragged out of device memory.

| | state lives in | barrier | launches for 800 steps | shared mem | max N |
|---|---|---|---|---|---|
| `global-sync` | device memory | `grid.sync()` | 1 | — | no limit |
| `gemm` | device memory | graph dependency | 1 graph launch | — | no limit |
| `block` | one block's shared memory | `__syncthreads()` | 1 | `9N` | ~25 800 |
| `cluster` | C blocks' shared memory | `cluster.sync()` | 1 | `9N/C` per block | ~412 000 with C=16 |
| `bit` | J slice + x, y in shared memory; sign bitmap in device memory | `grid.sync()` | 1 | `N·B/8 + 8·R·B + R·N/8` | ~4 000 at B=200 |

`bit` is the only one whose J traffic is per *solve* rather than per step, and
the only one that reads each J word once for every replica at once.  It
exists because on {-1,0,+1} couplings the product `J @ sign(x)` is an integer
identity (`n - 2·popcount(a XOR b)`), so nothing is rounded and nothing needs
a multiplier.

The first two are baselines, not proposals. `global-sync` is the obvious fused
kernel: keep everything in device memory and pay for a grid-wide barrier each
step. `gemm` stacks all `batch` sign vectors into an `N x B` matrix and lets
cuBLAS compute `J @ S` for every replica in one call, so each element of `J` is
read once per step and used `B` times — arithmetic intensity `B` instead of `1`,
on tensor cores. The state cannot stay on chip because the whole product must
finish before any replica advances. The existing `gemm` variant captures the
initial sign kernel plus all GEMM/update nodes into a CUDA Graph once, then
submits the complete run with one `cudaGraphLaunch`.

The shared-memory kernels keep all 800 steps fused in one persistent launch.

`J` is a single allocation in every dense path — the question was never how many
copies exist, but how many times each byte is fetched. In `block`, every
replica sweeps `J` row by row and whether two replicas share a fetch is left
entirely to L2. `cluster` exists to raise the `N` ceiling, not to change the
fetch pattern.

The CSR paths deliberately keep three distinct ownership schedules. `csr-row`
maps each work item to a sparse row and reuses its edge list across replicas.
On GPUs with cooperative launch support it executes all steps in one fused
kernel with a grid-wide step barrier; otherwise it falls back to a CUDA Graph
with double-buffered signs.
`csr-block` maps one persistent block to a replica and keeps `x`, `y`, and the
sign snapshot in shared memory whenever `9N` bytes fit, otherwise it falls back
to device-resident state. `csr-cluster` is the Hopper multi-block persistent
path for larger states. These schedules cannot share one kernel because their
cross-step barriers have different scopes.

`./bench` measures which regime you are actually in — see below.

## The idea

One replica is mapped to one CUDA block. The whole state vector lives in that
block's shared memory, so the global barrier that dSB needs after every step
collapses into `__syncthreads()` and an 800-step run is **one kernel launch**.

The price is shared memory: the state costs `9N` bytes per replica, which caps
`N` at about 25 800 on an H100-class GPU. Two things push past that:

* **FastHare** shrinks most instances below the cap before the solver sees them.
* **Thread block clusters** (`sm_90`) spread one replica over up to 16 blocks
  that read each other's sign array through distributed shared memory, lifting
  the state-capacity ceiling to roughly 412 000. Cluster 16 is Hopper-specific
  and is enabled with the non-portable cluster-size opt-in.

`make_plan()` picks between the two automatically.

---

## Layout

```
include/dsb/        common.hpp  kernels.cuh  solver.hpp  sparse_solver.hpp  sparse.hpp
                    bitfused.cuh  gemm.cuh  dense_capacity.hpp  reduction.hpp  qplib.hpp  gset.hpp
src/                kernels.cu  solver.cu  sparse_solver.cu  bitfused.cu  gemm.cu  dense_capacity.cu
                    sparse.cpp  reduction.cpp  qplib.cpp  gset.cpp
apps/               solve_qplib.cpp  solve_gset.cpp  export_fasthare.cpp  dense_scaling.cpp
                    selftest.cpp  bench.cpp  bit_dense_multi.cu
tools/              gpu_probe.cu  c2c_probe.cu
python/             reference_dsb.py  sb_schedule.py  suggest_dt.py  benchmark_public_dsb.py
                    benchmark_public_gset.py  benchmark_dense_scaling.py  fasthare_io.py
                    summarize_tts.py  summarize_fasthare_ablation.py  summarize_dense_scaling.py
                    generate_bipartite_scaling.py  generate_regular_scaling.py  check_contamination.py
scripts/            lib_guard.sh  run_gh200_comparison.sh  run_gh200_gset_comparison.sh
                    run_dense_scaling.sh  run_scaling.sh  run_paper_suite.sh
                    run_bit_multi.sh  run_bit_mattis.sh
run_K2000.sh  run_Gset.sh  run_qplib.sh  run_large_scale.sh     benchmark drivers (repository root)
run.md              how every experiment was run
results/            every result directory (see results/README.md)
tests/              unit tests for the Python helpers
third_party/fasthare/
data/qplib/          QPLIB data and solution files
data/Gset/           G-set data directory and best_known.csv
data/K2000/          WK2000_1.rud and its target
data/scaling/        generated bipartite scaling graphs (gitignored)
docs/               code review, shared-memory analysis, bit kernel, dense scaling
                    protocol, comparison protocol, bit_dense_multi
```

### Renamed from the previous version

| was | is |
|---|---|
| `CDSB` (class) | `dsb::Solver` |
| "CDSB" (everywhere) | dSB — the algorithm is Goto's *discrete* SB, not a continuous-time variant |
| `cdsb_fastshare.{h,cu}` | `dsb/kernels.cuh`, `src/kernels.cu`, `src/solver.cu` |
| `fastshare` | gone; it was a typo of *FastHare*, and only the reducer is FastHare |
| `cdsb_fused_dense_fastshare_kernel_fp16` | `step_block_kernel<T>` |
| `cdsb_fused_run_fp16` / `_fp32` | `launch_steps` (overloaded on storage type) |
| `cdsb_energy_kernel_fp16` | `energy_kernel<T>` |
| `fasthare_api::fasthare_reduction` | `dsb::reduce_ising` |
| `cdsb_fasthare_qplib.cpp` | `apps/solve_qplib.cpp` |
| `dSB_original_python.py` | `python/reference_dsb.py` |
| `CDSB_CUDA_CHECK` | `DSB_CUDA_CHECK` |

The old `optimized_kernel.py`, `cooperative_batched_fused_code.py` and
`MultiRowSharedFused.py` are **not** carried over. They were earlier rungs of
the same ladder and two of them synchronised incorrectly across blocks. Keep
the originals around if you want them for the ablation table in the paper, but
do not build on them.

---

## Build and run

```bash
make                     # solvers, dense_scaling, selftest, bench, bit_dense_multi
make probe               # gpu_probe, c2c_probe

./gpu_probe              # what this GPU actually allows -- run this first
./selftest               # kernels vs the CPU reference; run after any change
./bench 8192 132 100     # which variant wins, and why

./solve_qplib data/QPLIB_3506.qplib
./solve_qplib --csv --plan-only data/*.qplib        # the size table
./solve_qplib --variant=block --l2 data/QPLIB_3506.qplib
./solve_gset --variant=gemm --batch=200 --steps=800 data/K2000/data/WK2000_1.rud

# Same-instance 2x2 comparison (public/GPU x FastHare off/on) on this GH200.
python3 -m pip install -r requirements-baseline.txt  # after CUDA PyTorch
scripts/run_gh200_comparison.sh data/QPLIB_3506.qplib [TARGET_OBJECTIVE]

# G-set / K2000 Max-Cut. The comparison runs both FastHare modes for the
# public baseline and every requested existing GPU schedule.
./solve_gset data/Gset/data/G1
./solve_gset --variant=csr-row --precision=fp32 data/Gset/data/G22
./solve_gset --variant=csr-cluster --cluster=8 --precision=fp32 data/Gset/data/G22
scripts/run_gh200_gset_comparison.sh data/Gset/data/G22 [TARGET_CUT]

# Dense GH200 capacity scaling and then the complete paper suite.
GPU_ID=1 scripts/run_dense_scaling.sh
scripts/run_paper_suite.sh

# Dense +-1 couplings at one bit per coupling on two GH200s (HBM first, rest in Grace memory).
./bit_dense_multi --n 8192 --gpus 2 --steps 100 --check --hbm-rows 1024   # bit-for-bit vs host reference
./bit_dense_multi --n 3000000 --gpus 2 --dry-run                           # memory plan only
GPUS=2 N_LIST="500000 1000000 2000000 2500000 3000000" scripts/run_bit_multi.sh
GPUS=2 N_LIST="500000 1000000 2000000 2500000 3000000" scripts/run_bit_mattis.sh   # planted optimum, PASS/FAIL per run
```

The three benchmark drivers write to `results/run_<suite>_result/` by default
(`RESULT_ROOT` overrides); every other script writes under `results/` too.

All benchmark scripts select the second physical GPU by default
(`GPU_ID=1`, exposed to both C++ and Python as `cuda:0`). Override explicitly,
for example `GPU_ID=0 scripts/run_dense_scaling.sh`.

Needs CUDA 12.0+ (thread block clusters), cuBLAS and Eigen 3. Override paths
with `make EIGEN_INC=... CUDA_HOME=...`.

`solve_qplib` options: `--alpha`, `--batch`, `--steps`, `--precision=fp16|fp32`,
`--variant=csr-row|csr-block|csr-cluster|block|cluster|global-sync|gemm|auto`,
`--cluster=N` (0 = auto), `--l2`,
`--seed`, `--repeat`, `--target`, `--no-reduction`, `--no-gauge`, `--csv`,
`--plan-only`.

`solve_gset` accepts the same performance and reporting options except
`--no-gauge`, which is unnecessary because G-set has no linear field. Its
reported objective is the weighted cut value, including negative edge weights.
Use `--variant=csr-row|csr-block|csr-cluster --precision=fp32`
for the CSR-native paths and
`--no-reduction` for pure implementation scaling. CSV output additionally
reports time per step, edge updates per second, coupling storage, and total GPU
allocation. One edge update means one undirected edge processed for one replica
in one dSB step.

For the existing `gemm` variant, `--batch` fixes the GEMM/graph shape and
`--steps` fixes the captured node sequence. Capture and instantiation occur
before the CUDA timing events. A second call on the same solver reuses the same
executable graph; changing either shape causes a one-time recapture.

The comparison scripts pass `BATCH` and `STEPS` once when each solver is
created (defaults: 200 and 800), execute unmeasured warm-up work first, and
only then emit measured repetitions. They do not change either graph-shape
parameter inside a measured run.

### FastHare experiment matrix

Both QPLIB and G-set comparison scripts now produce the complete experiment
matrix without adding a variant name:

| reduction | public baseline | dsb-gpu |
|---|---|---|
| off | raw public dSB | raw GPU dSB |
| on | FastHare + public dSB | FastHare + GPU dSB |

`export_fasthare` serializes the exact residual Hamiltonian, sign vector and
spin map produced by the in-tree reducer. The public baseline consumes that
file, so the two reduced solvers see the same reduced problem. Every result is
lifted back and evaluated on the original QPLIB objective or original weighted
G-set cut.

The primary pipeline time is defined exactly as

```text
total_s = preprocess_s + solver_s + reconstruction_s
```

where reconstruction includes selecting/recovering the winning assignment and
applying the FastHare map. `evaluation_s` checks the recovered assignment on
the original problem and is reported separately. `wall_s` additionally keeps
model/setup, transfer, evaluation, and other measured process overhead visible.
The scripts write `summary.csv`, `summary_solver.csv`, `summary_wall.csv`, and
`fasthare_ablation.csv`; the last file directly labels the raw-solver,
post-FastHare-solver, and end-to-end comparisons.

The 71 target values in `data/Gset/best_known.csv` are best-known results, not
general claims of proven optimality. See `data/Gset/README.md` for sources and
the newer G63/G72/G77/G81 values.

### CSR paths

G-set input is parsed directly into a symmetric CSR matrix. `SparseSolver`
keeps the coupling in CSR and supplies three schedules: row ownership across a
replica batch, a persistent block with a thread per row, and a persistent
Hopper cluster. It never constructs an `N x N` matrix. FastHare also has a CSR
input/output wrapper.

The former sparse bipartite scaling driver remains as `scripts/run_scaling.sh`
for an optional appendix, but it is no longer the main paper scaling figure.

### Main dense GH200 capacity scaling and the placements beyond HBM

`scripts/run_dense_scaling.sh` generates deterministic symmetric FP32 matrices
with approximately 100% nonzero high-entropy weights. It disables FastHare and
CSR. Defaults are `N=10k,20k,40k,80k,120k,160k,200k,240k,300k`, batch 1, 50 steps,
three repeats, seed 42, and physical GPU 1.

`--matrix-memory=auto` queries currently free HBM. If the matrix plus state is
within 85% of free HBM it uses `cudaMalloc`; otherwise the whole matrix goes
into Grace memory: an anonymous `mmap` bound to the GPU's NUMA node
(`GRACE_NUMA_NODE`), pinned with `cudaHostRegister`, and streamed every step
through a 2 GiB HBM staging buffer inside the same CUDA Graph as the products
(371 GB/s measured; the earlier driver-managed version reached 80--91 GB/s).
`--matrix-memory=hybrid` fills HBM with the first rows and stages only the
remainder. The script records the placement, bytes in each tier, generation
time, GPU time/step, dense interactions/s, effective streamed bandwidth, and
structured failure status (`oom`, `launch-limit`, `timeout`). `tools/c2c_probe.cu`
measures the raw C2C read bandwidth of such an allocation.

Run a smaller smoke test first:

```bash
GPU_ID=1 N_LIST="20000 40000" STEPS=5 REPEATS=1 \
  scripts/run_dense_scaling.sh
```

Then run the paper sweep:

```bash
GPU_ID=1 scripts/run_dense_scaling.sh
```

On a dual-GH200 NUMA system, first inspect `nvidia-smi topo -m` and
`numactl --hardware`. If physical GPU 1 is local to Grace NUMA node 1, use
`GPU_ID=1 GRACE_NUMA_NODE=1 scripts/run_dense_scaling.sh`. Do not assume that
the numeric GPU and NUMA identifiers match; the script records the topology.

The public PyTorch baseline uses its normal HBM allocator. The defensible paper
claim is therefore "among the evaluated implementations, only the adaptive
GH200 implementation completed the largest dense instances", not that no
possible Grace-aware baseline could ever be written.

The comparison script uses the third-party open-source
[`simulated-bifurcation`](https://github.com/bqth29/simulated-bifurcation-algorithm)
package in discrete mode. It is a runnable public baseline, **not** Toshiba's
official implementation. See [docs/public-dsb-comparison.md](docs/public-dsb-comparison.md)
for the exact timing and TTS protocol.

---

## Reading `./bench`

```
  variant      C    L2     ms/step   J fetches/step  speedup
  -----------------------------------------------------------
  global-sync  1    off      98.100           231.0    1.00x
  gemm         1    off       0.640             1.1  153.28x
  block        1    off      12.340            29.1    7.95x
  block        1    on       12.201            28.8    8.04x
  ...
```

(Illustrative shape only — the real numbers come from your GPU.) Speedups are
quoted against `global-sync`, the rung the shared-memory design was introduced
to replace.

The column that decides everything is **J fetches/step**, estimated as
`(time per step) x (measured read bandwidth) / sizeof(J)`:

* **near 1** — every byte of `J` was fetched once and reused by all concurrent
  replicas. L2 is already doing for free what a GEMM formulation would do
  explicitly, and the shared-memory design wins outright. Stop here.
* **large** — the replicas are out of phase and each is pulling its own copy of
  `J` through the memory system. This is where `gemm` earns its launch overhead.

`gemm` should sit near 1 by construction — it is the control. If `block` is
also near 1, L2 is already doing the sharing for free and the shared-memory
design wins outright. If `block` is an order of magnitude higher while `gemm`
is near 1, the sharing is worth buying explicitly.

It is an estimate, not a hardware counter: it assumes the kernel is bandwidth
bound, which stops being true at small `N`. Confirm with
`ncu --metrics dram__bytes.sum,lts__t_sector_hit_rate.pct ./solve_qplib ...`
before putting the number in a paper.

`--l2` pins as much of `J` in L2 as the device allows. The carve-out is capped
at a fraction of an already small cache — tens of megabytes against a `J` that
is gigabytes — so expect it to matter only for small instances. `gpu_probe`
prints the cap; `bench` runs every variant both ways so you can see where the
line falls.

---

## N ceilings

State is `x[N]` and `y[N]` in FP32 plus `sign[N]` as `int8`, so `9N` bytes.

| | shared / block | max N |
|---|---|---|
| single block | 227 KB | ~25 800 |
| cluster of 2 | 2 × 227 KB | ~51 600 |
| cluster of 4 | 4 × 227 KB | ~103 000 |
| cluster of 8 | 8 × 227 KB | ~206 000 |
| cluster of 16 | 16 × 227 KB | ~412 000 |

`gpu_probe` prints the real numbers for your GPU. A cluster is only used when
the plain block kernel does not fit.

**Fitting is not the same as being fast.** Every block re-reads all of `J` on
every step, and the inner loop does one multiply-add per element of `J` — the
arithmetic intensity is O(1), so past roughly `N = 50 000` the run is pure HBM
streaming. At `N = 100 000`, `J` in FP16 is 20 GB and a single step costs about
6 ms. Use `--plan-only` to see where an instance lands before launching it.

---

## What changed relative to the original code

**Bugs fixed**

* **Shared memory was never opted in.** Without
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySizeBytes, ...)`
  a kernel is capped at 48 KB of dynamic shared memory regardless of hardware,
  so the old kernel died above `N = 2457` on an H100 — not `N = 11600` as its
  README claimed. `launch_steps` now opts in.
* **The bias-node gauge was dropped on one path.** The standard form folds the
  external field into an extra spin, so the recovered assignment is only
  defined up to that spin's sign. The fully-reduced branch multiplied through
  by it; the normal branch did not, which mirrors roughly half the solutions
  and scores the wrong objective. `lift_solution()` now applies it on both
  paths. `--no-gauge` restores the old behaviour so you can A/B it.
* `std::clock()` (CPU time, meaningless here) replaced with
  `std::chrono::steady_clock`.

**Made smaller and faster**

* The `x1`/`y1` ping-pong buffers are gone. They were protecting a Jacobi
  sweep that was already protected: the coupling term reads the `sign`
  snapshot, never `x`, so updating `x` in place is safe. With `sign` also
  dropped from FP32 to `int8`, the state went from `20N` to `9N` bytes — the
  reachable `N` went up 2.2× and each step lost a copy loop and a barrier.
* `J` rows are padded to a multiple of 8 elements and read with 16-byte vector
  loads, so a warp pulls 512 B per instruction instead of 64 B.
* `xi` is computed from the original FP64 matrix instead of the quantised copy,
  so switching `--precision` no longer changes the dynamics as a side effect.
* The pump schedule is FP32 on the device. In the Python reference it now lives
  on the GPU too — it used to be a CPU tensor, which cost an implicit transfer
  and a sync on every step and inflated any speedup measured against it.

**Added**

* `selftest` — checks FP32 against a double-precision CPU reference, checks
  FP16 on energy, and checks the CSR and cluster paths.
* `bench` — runs every variant with and without L2 pinning and estimates how
  many times `J` was actually fetched per step.
* The existing `gemm` variant captures its full step loop into a CUDA Graph;
  no additional variant name or CLI mode is introduced.
* `global-sync` and `gemm` baselines, so the internal ablation ladder is
  complete: the local PyTorch reference -> global-sync -> gemm -> block ->
  cluster. The separate public-package comparison is described
  below.

---

## On comparing against published work

The two baselines here are *algorithmic*, not reimplementations of any
particular paper: `global-sync` is the obvious fused kernel and `gemm` is the
standard dense formulation. They are deliberately not labelled with anyone's
name.

Reimplementing a published kernel from a description — or worse, from memory —
and then reporting it as that author's method is how comparisons end up
misrepresenting the work they cite. For a paper, compare on the instances the
literature already reports: K2000 and the G-set, against the numbers printed in
those papers, at matched time budgets. If you want a specific method implemented
faithfully, work from its paper and its released code, not from a summary.

For a runnable same-hardware comparison, this repository now wraps version
2.0.0 of the public `simulated-bifurcation` PyTorch package and feeds it the
same problem instance. Run `scripts/run_gh200_comparison.sh` for QPLIB or
`scripts/run_gh200_gset_comparison.sh` for G-set; each records the GH200
software stack and produces raw/summary CSV, including TTS99 when a target is
provided. Use FP32 for the primary cross-implementation comparison and keep
this solver's FP16 result as a separate ablation.
* `gpu_probe` — reports the real shared-memory budget, cluster support and a
  DSMEM smoke test.
* `--csv` / `--plan-only` on either solver, which emit the
  original-N / reduced-N / fits-in-one-block table directly.

---

## Verification

Run `make -j` and `./selftest 512 4 200` after pulling the code. The self-test
checks the retained CSR schedules and the dense block/cluster paths.
The archive was prepared in an environment without `nvcc` or a CUDA GPU, so
the CUDA build and runtime checks must be completed on the target GH200.

The cluster path is the part to distrust most. Check in this order:

1. `./gpu_probe` — does `DSMEM map_shared_rank` report `OK`?
2. `./selftest 512 4 200` — the CSR row/block checks and `C=2`/`C=4` results
   must match their reference results.
3. `./selftest` also checks `global-sync` and `gemm` against the block kernel;
   the cuBLAS call's operand order and leading dimensions are the part most
   likely to be wrong, and that check is what catches it.
4. `./bench` — look at `J fetches/step` before optimising anything else.
5. Only then try a large instance.

Hardware figures in this README (227 KB per block, 228 KB per SM) come from
memory and were not checked against NVIDIA's documentation. `gpu_probe` prints
the authoritative values — use those in the paper.

---

## License

This repository is released under the MIT License (see [`LICENSE`](LICENSE)).
`third_party/fasthare/` is the FastHare reducer of Thang Dinh and Phuc Thai
(https://github.com/thang-dinh/FastHare), also MIT-licensed; its original
copyright notice applies to those files. The QPLIB, G-set and K2000 instances
are not redistributed here; see `data/README.md` and `data/Gset/README.md` for
their sources.

## Repository hygiene

`.gitignore` excludes everything `make` produces (object files, the eight
binaries), profiler output, Python caches, the downloaded/generated instances
under `data/`, and `results/` (only its README is versioned; the raw
measurements are archived with the paper). `make clean` removes the build
products; nothing under `src/`, `apps/`, `scripts/` or `docs/` is generated.
