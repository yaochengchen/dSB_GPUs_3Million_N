# Public dSB comparison on GH200

## What this baseline is

`python/benchmark_public_dsb.py` and `python/benchmark_public_gset.py` run
version 2.0.0 of the MIT-licensed
[`simulated-bifurcation`](https://github.com/bqth29/simulated-bifurcation-algorithm)
package in `mode="discrete"` on CUDA. It is a public, independently maintained
PyTorch implementation of dSB. It is **not** Toshiba's unreleased production
code and must not be labelled that way in a paper.

Each Python script and its corresponding C++ solver use the same input file,
objective convention, agent count, step count, target, and seed sequence. All
emit one row per batched run in a shared CSV schema. For G-set, the objective
is the weighted cut value and the default target is read from
`data/Gset/best_known.csv`.

FastHare is an on/off preprocessing column, not a solver variant. For reduced
runs, `export_fasthare` writes the exact residual Hamiltonian and reconstruction
map used by `dsb-gpu`; the Python baseline reads the same file. Thus differences
between the reduced solver timings cannot come from different reduction
decisions.

## Quick run

Install a CUDA-enabled PyTorch build suitable for the GH200, then:

```bash
python3 -m pip install -r requirements-baseline.txt

# Without a known target: compare time and solution quality.
scripts/run_gh200_comparison.sh data/QPLIB_3506.qplib

# With a known target: also calculate batch success probability and TTS99.
REPEATS=20 BATCH=200 STEPS=800 \
  scripts/run_gh200_comparison.sh data/QPLIB_3506.qplib -123456

# G-set: target is looked up automatically; a second argument overrides it.
REPEATS=20 BATCH=200 STEPS=800 \
  scripts/run_gh200_gset_comparison.sh data/Gset/data/G22
```

Outputs are written below `results/<UTC timestamp>/`:

- `environment.txt`: GPU, driver, CUDA, PyTorch and package versions;
- `dsb_gpu.csv`: every requested GPU variant with FastHare off and on;
- `public_dsb_reduction0.csv` and `public_dsb_reduction1.csv`: raw and reduced
  public-baseline runs;
- `fasthare_reduction.txt`: the shared residual Hamiltonian and lift map;
- `raw.csv`: all four experiment cells combined;
- `summary.csv`: primary pipeline time, objective, success rate and TTS99;
- `summary_solver.csv` and `summary_wall.csv`: diagnostic timing views;
- `fasthare_ablation.csv`: the three direct publication comparisons;
- `gpu_probe.txt` and `selftest.txt`: short hardware/correctness checks.

Set `SKIP_SELFTEST=1` to omit the short self-test. Other useful environment
variables are `REPEATS`, `BATCH`, `STEPS`, `BASE_SEED`, `CUSTOM_VARIANTS`, `PRECISION`,
`REDUCTION_MODES`, `FASTHARE_ALPHA`, `OUT_DIR` and `PYTHON`.
`GPU_ID=1` is the default, so every C++ and Python process uses the second
physical GPU (visible inside the process as `cuda:0`).
For the multi-mode scripts, use `CUSTOM_VARIANTS`. Both scripts default to
`REDUCTION_MODES="0 1"`. The older G-set-only `REDUCTION=0|1` variable is still
accepted as a single-mode compatibility switch.

`BATCH` and `STEPS` are shape parameters (defaults 200 and 800). They are fixed
when a solver/graph is created. Warm-up is unmeasured, and the CUDA Graph GEMM
path captures/instantiates before its CUDA timing interval; measured launches
reuse the fixed graph shape.

## Required experiment separation

`fasthare_ablation.csv` contains one row per GPU variant for each applicable
comparison:

| label | baseline | candidate | purpose |
|---|---|---|---|
| `solver-no-reduction` | public dSB | GPU dSB | CUDA implementation speedup |
| `solver-after-fasthare` | FastHare + public dSB | FastHare + GPU dSB | reduced-solver speedup |
| `end-to-end-fasthare-pipeline` | public dSB | FastHare + GPU dSB | complete pipeline benefit |

The same file carries preprocessing time, reconstruction time, reduction ratio,
original/reduced sizes, objective delta, median-time speedup, and TTS99 speedup.
All objective values are calculated after lifting on the original problem.

## What to report

Use FP32 for the main apples-to-apples comparison: the public package accepts
FP32/FP64, while this repository also has an FP16 storage path. Report the FP16
path separately as an internal precision/performance ablation.

For every result, report:

1. GH200 model, driver, CUDA, PyTorch and baseline package version;
2. original and FastHare-reduced problem size;
3. agents, integration steps, precision and early-stopping policy;
4. median `total_s`, `wall_s`, and objective distribution;
5. target definition, batched success probability and TTS99 when a defensible
   target is available.

For a batch success probability `p` and median batch wall time `t`, the summary
uses:

```text
TTS99 = t * log(0.01) / log(1 - p)
```

If `p=1`, TTS99 is one batch time; if `p=0`, it is infinite. Five repeats are
enough for a quick engineering check, but publication-quality success rates
need more independent runs.

## Timing definition and caveats

- `total_s` is the primary algorithmic pipeline number and is defined as
  `preprocess_s + solver_s + reconstruction_s`. This is the number used by
  `summary.csv` and its default TTS99 calculation.
- `preprocess_s` is the measured FastHare reduction time. The exported value is
  charged to both reduced implementations even though the residual file is
  created once for reproducibility.
- `reconstruction_s` includes winning-state recovery and the FastHare lift.
  `evaluation_s` evaluates the lifted assignment on the original problem and
  is separate, so it cannot improve an algorithmic TTS result.
- `wall_s` includes measured setup/model construction, solving,
  reconstruction, evaluation, transfers, and other in-process overhead. Python
  import and the initial input text read remain outside the repeated section.
- `solver_s` is useful diagnostically, but it is not perfectly symmetric:
  `dsb-gpu` obtains it from CUDA events around its dynamics, whereas the public
  package is timed around its public `Ising.minimize` call and necessarily
  includes package-side tensor preparation.
- Early stopping is disabled by default so `STEPS` means the same fixed budget.
- A seed aligns the run schedule, not the initial states: the C++ and PyTorch
  random-number generators are different.
- Do not compare a single-GH200 measurement directly with published FPGA or
  multi-GPU timings as if only the algorithm differed. Quote those as external
  reference points with their hardware and stopping rules intact.
- G-set comparison rows explicitly identify `storage=csr|dense` and whether
  FastHare was enabled. The optional `run_scaling.sh` path remains CSR-only.
  The main paper capacity experiment is `run_dense_scaling.sh`: it disables
  FastHare and CSR, crosses the HBM boundary, and records HBM/Grace placement,
  time per step, throughput, and structured failure reasons.
