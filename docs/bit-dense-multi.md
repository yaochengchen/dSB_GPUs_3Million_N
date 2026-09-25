# Dense ±1 dSB at one bit per coupling on one or two GH200s

`apps/bit_dense_multi.cu` is the second implementation of the dense ±1 form of the
`bit` path. The benchmark kernel (`src/bitfused.cu`, `docs/bit-kernel.md`) keeps
the bit planes and a replica tile in shared memory and therefore stops at
n ≈ 2000 for B = 512. This program keeps the packed matrix in device memory,
integrates one replica, and splits the matrix over the two GH200s of the
system, so that the storage advantage of the format (1/8 byte per coupling)
can be used where it matters: it reaches n = 3,000,000 (1.125 TB of couplings)
on two GPUs.

## What it computes

The same dSB step as every other path (Eqs. 3–5 of the paper), for a random
symmetric J ∈ {±1}^{n×n}:

```
A_i = Σ_j J_ij s_j = n − 2·popc(J_i XOR S) − s_i       (diagonal stored as +1)
y  += (−(δ − p_k) x + ξ A) Δt ;  x += Δt y δ ;  |x| > 1 → x = sgn x, y = 0
```

`ξ = 0.5/√n` (the usual `0.5·√(n−1)/‖J‖_F` for a ±1 matrix), `Δt = 1.1` by
default, `p_k = k/(K−1)`, initial x, y uniform in (−0.01, 0.01) from the seed.
The product is an exact integer (Proposition 1(iii) of the paper); the update
uses explicit round-to-nearest intrinsics on the device and is compiled with
`-fmad=false` / `-ffp-contract=off` on both sides, so the trajectory is
bit-for-bit independent of the number of GPUs and of the placement, and
`--check` compares the final state with a host reference (n ≤ 50,000). Verified 2026-09-25 at n = 8192, K = 100: one GPU, two GPUs, and two GPUs with `--hbm-rows 1024` all PASS with the same checksum.

## Memory plan

* Row i is `ldw` 32-bit words, `ldw` rounded up to a multiple of 4 (128-bit
  loads); bit 1 means +1. J is generated on the GPU from a hash of (i, j) —
  nothing is ever written to disk.
* Rows [0, n/2) go to GPU 0 and [n/2, n) to GPU 1 (`--gpus 1` keeps all rows
  on one GPU). On each GPU the first `head` rows are a `cudaMalloc` block
  (`--hbm-fraction 0.85` of free HBM after state, bitmaps and the staging
  buffer), and the remaining `tail` rows are an anonymous `mmap` bound with
  `mbind` to the GPU's own NUMA node (`cudaDevAttrHostNumaId`, overridable with
  `--numa 0,1`) and pinned with `cudaHostRegister`. This is the hybrid
  placement of `src/dense_capacity.cu`; `--placement grace` forces everything
  into Grace memory, `--placement hbm` refuses to run if the part does not fit,
  `--hbm-rows R` caps the HBM part (used by `--check` to exercise staging at
  small n).
* Every step the tail is copied in 2 GiB blocks (`kStageBytes`) into an HBM
  staging buffer and consumed there by the popcount kernel; copy and kernel are
  serialised on one stream, as in the gemm path.
* `popc_rows`: one warp per 4 rows (`kRowsPerWarp`), lanes stride over the
  row in `uint4`, the sign bitmap is read through the read-only cache and the
  matrix with streaming loads (`__ldcs`, it is touched once per step).
* `update_rows` writes the new sign bits of its rows into the bitmap with a
  warp ballot; the two GPUs then exchange their bitmap segments (n/8 bytes
  each) with `cudaMemcpyPeerAsync` over NVLink and wait on each other's copy
  event. That exchange is the only inter-GPU traffic.
* Two bitmaps (`v[0]`, `v[1]`) are double-buffered across steps; the energy
  `E = −½ Σ_i s_i A_i` and an FNV checksum of the final spins are computed
  after the last step. Repeat 0 is a warm-up and is not reported.

`--dry-run` prints the plan (rows per GPU, HBM/Grace split, free memory of each
node) without allocating; the CSV row then carries `status=planned` or
`does-not-fit`.

## Mattis instances

`--instance mattis` generates `J_ij = ξ_i ξ_j` from a hidden random
ξ ∈ {±1}^n derived from `--seed`. Under the gauge `s_i → ξ_i s_i` this is the
ferromagnet, so the ground states are s = ±ξ with energy exactly
−n(n−1)/2; the run prints `expected` next to the energy and the CSV column
`found` is 1 when they agree. Changing the seed changes both ξ and the initial
state, so several seeds are several independent trials of one problem;
repeating a seed reproduces the same run exactly (`REPEATS=1` in the script).
It is an implementation check — data path, staging, exchange, energy — not a
statement about the optimisation power of dSB on hard problems of this size.
A harder planted family is one line away: flip the sign of ξ_i ξ_j with
probability p in `jbit()`.

## Scripts and outputs

```
scripts/run_bit_multi.sh    GPUS=2 N_LIST="500000 1000000 2000000 2500000 3000000" STEPS=50 REPEATS=3
scripts/run_bit_mattis.sh   same n, SEEDS="42 43 44", REPEATS=1, prints a PASS/FAIL line per run (also saved to check.txt), non-zero exit on any FAIL
```

Both write `results/bit_{multi,mattis}_<UTC stamp>/raw.csv` (header from
`--csv-header`), `environment.txt` (nvidia-smi, topology, `numactl --hardware`)
and one `n<N>[_s<seed>].log` per run with the plan, the per-repeat timing and,
for the profiled step, the time of the HBM rows, the Grace rows, the update
and the exchange on each GPU. `dsb_analysis` reads them into
`out/tables/bit_multi_summary.csv`, `tab_bit_multi.tex` and `fig9_bit_multi`.

## Measured (2026-09-25, two GH200 144G, K=50, B=1)

| n | matrix | HBM + Grace | s/step | aggregate GB/s | Mattis |
|---:|---:|---|---:|---:|---|
| 500,000 | 31 GB | 31 + 0 | 0.0035 | 8952 | 3/3 |
| 1,000,000 | 125 GB | 125 + 0 | 0.0139 | 8991 | 3/3 |
| 2,000,000 | 500 GB | 254 + 246 | 0.407 | 1229 | 3/3 |
| 2,500,000 | 781 GB | 254 + 527 | 0.844 | 926 | 3/3 |
| 3,000,000 | 1125 GB | 254 + 871 | 1.40 | 803 | 3/3 |

* HBM-resident: 4.49 TB/s per GPU (92 % of the HBM3e peak).
* Hybrid: `T = M_H/8.97 TB/s + M_G/646 GB/s` reproduces the three points within
  2 %. Per GPU the Grace share streams at 317 GB/s (GPU 0) and 352 GB/s (GPU 1)
  — 87 % of the 371 GB/s of the single-GPU gemm path, the rest being the
  serialisation of each staged copy with the popcount kernel; the step is set
  by the slower link and GPU 1 waits ~138 ms per step at n = 3M.
* Bitmap exchange: 0.1 ms per step at n = 3M (375 KB). Generation + pinning:
  81 s at n = 3M, not part of the step time.
* n = 3M uses 94 % of the combined 2 × (0.85·142.5 + 480) GB budget; the
  analytic limit of this format on two GPUs is n ≈ 3.1 M.

## Not done

* B > 1: the bitmap and the popcounts would have to be kept per replica; then
  B trajectories would share each pass over the matrix as in the gemm path.
  Only worth doing for TTS statistics on hard instances.
* Overlapping the staging copies with the popcount kernel (≤ 8 % of the step).
* The ternary (sparse) bit form and INT8 beyond HBM.
