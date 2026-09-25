# results/

Every measurement of the paper lives here; `dsb_analysis/run_all.py --root <this repo>` reads nothing else.
Directories are never edited by hand; when a configuration was re-run, both copies stay and the analysis
takes the latest one (benchmarks: start time in `environment.txt`; dense scaling: version number in the
directory name; bit runs: the time stamp in the directory name).

| directory | produced by | what it holds |
|---|---|---|
| `run_K2000_result_v5/` | `run_K2000.sh` | K2000, B=512, K=200..3200, every variant (15 up to K=1600, the fixed set of 8 at K=3200) + matched SB 2.0.0 |
| `run_Gset_result_v5/`, `run_Gset_result_v5_supp/` | `run_Gset.sh` | 12 + 4 G-set instances, B=512, K=200..3200, the fixed G-set variant set + matched SB |
| `run_qplib_result_v5/`, `run_qplib_result_v5_supp/` | `run_qplib.sh` | 19 QPLIB instances, B by n, K=200..3200; `_supp` adds `auto`, `bit`, `gemm:fp32:notf32` |
| `run_Gset_int8_v6_supplement/`, `run_Gset_int8_v7/` | `run_Gset.sh` | `gemm:int8` on G43 48 55 58 60 63 64 (v6) and the G66 re-run (v7; the v5_supp G66 INT8 timing was disturbed) |
| `run_K2000_result_v7_B16/`, `_B64/`, `_B128/` | `run_K2000.sh` | K2000 batch sweep: `bit`, `gemm:fp16`, `gemm:int8`, `gemm` (TF32) + matched SB |
| `run_Gset_agents_v5_supp/` | loop in `run.md` §3.4 | G55 60 63 70 72 81 at B=1024..8192, `auto` and `csr-block`, no baseline |
| `run_Gset_library_v5_supp/`, `run_qplib_library_v5_supp/` | `run_Gset.sh` / `run_qplib.sh` with `SB_SCHEDULES=library` | 66 G-set instances (`csr-row`) and 19 QPLIB (`csr-block`) at K=800 against SB 2.0.0 library defaults |
| `scaling_sparse_v5_s200/`, `_s400/` | `scripts/run_scaling.sh` + `python/benchmark_public_gset.py` | degree-100 bipartite graphs, n=10k..200k, `csr-row`, B=200; PyTorch baseline up to n=50k |
| `dense_scaling_v5/` | `scripts/run_dense_scaling.sh` | dense random real J, n=20k..300k, B=1/8/64, FP32+FP16, `gemm` in HBM + PyTorch baseline |
| `dense_scaling_v9_grace_{fp16,fp32}/` | `scripts/run_dense_scaling.sh`, `MATRIX_MEMORY=auto` | whole matrix in pinned Grace memory, staged through HBM (the numbers in the paper) |
| `dense_scaling_v9_hybrid_{fp16,fp32}/` | `scripts/run_dense_scaling.sh`, `MATRIX_MEMORY=hybrid` | HBM filled first, remaining rows in Grace memory (the numbers in the paper) |
| `dense_scaling_v5_supp_grace*/`, `dense_scaling_v7_grace_fp16/`, `dense_scaling_v8_*/` | `scripts/run_dense_scaling.sh` | earlier Grace placements: driver-managed memory (80--91 GB/s) and the first staging version; superseded by v9, kept as a record |
| `bit_multi_<stamp>/` | `scripts/run_bit_multi.sh` | dense ±1 bit path on two GH200s, random ±1 J, n=0.5M..3M, K=50, B=1, 3 repetitions |
| `bit_mattis_<stamp>/` | `scripts/run_bit_mattis.sh` | same path on planted Mattis instances, 3 seeds per n; `found`=1 when the energy equals -n(n-1)/2 |

File conventions inside a benchmark run directory `<instance>_<B>_<K>/`: one `dsb_gpu_<variant>-<precision>_reduction0.csv`
per dsb-gpu entry, `public_dsb_<dtype>_<schedule>_reduction0.csv` per baseline entry, `environment.txt` (start
time, dt, settings), `run_status.csv`, `.done` when the directory is complete; `raw.csv` and `summary*.csv` are the
runner's own concatenation and are not used by the analysis. A zero-byte `.csv` means the entry failed or the copy
was truncated; `run_all.py` warns about directories that are `.done` but unreadable.
