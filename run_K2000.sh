#!/usr/bin/env bash
set -Eeuo pipefail

# Run the K2000 step sweep from the dsb-gpu repository root.
# The published target cut is fixed at 33337.
# Output layout: results/run_K2000_result/K2000_<batch>_<steps>/  (RESULT_ROOT overrides the root)
#
# Examples:
#   ./run_K2000.sh
#   REPEATS=5 STEPS_LIST="200 400" ./run_K2000.sh
#   REDUCTION_MODES="0 1" ./run_K2000.sh   # bring FastHare back

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-1}"
# BATCH=512: the largest batch at which the `bit` plan for N=2000 (16 rows per
# block, ~209 KB dynamic smem) still fits GH200's 227 KB opt-in limit with 125
# co-resident blocks on 132 SMs; hard cap is ~560.  Also a multiple of 4 for
# int8 gemm.  gemm/csr-row/public PyTorch state at this batch is < 1 GB even
# at N=20000, so device memory is not a concern.
BATCH="${BATCH:-512}"
# REPEATS=50: p_batch is resolved to 1/REPEATS, so TTS99 needs >= 50 runs.
REPEATS="${REPEATS:-50}"
WARMUP="${WARMUP:-1}"
PRECISION="${PRECISION:-fp32}"
FASTHARE_ALPHA="${FASTHARE_ALPHA:-0.2}"
# FastHare is off by default (only reduction=0).  REDUCTION_MODES="0 1" restores
# the ablation; export_fasthare and fasthare_ablation.csv only run for mode 1.
REDUCTION_MODES="${REDUCTION_MODES:-0}"
# variant[:precision[:notf32]]; see scripts/run_gh200_gset_comparison.sh.
# Default = every variant x precision solve_gset implements:
#   auto          dispatcher only (bit / csr-row / gemm); rows labelled auto->X
#   bit           XOR+POPC bitplane kernel, J on chip (N <= ~2000 at batch 512)
#   csr-row csr-block csr-cluster   sparse kernels (fp32 only)
#   block cluster global-sync       dense persistent kernels (fp32 and fp16)
#   gemm          cuBLAS fp32 storage + TF32;  gemm:fp32:notf32 = CUDA cores
#   gemm:int8     cuBLAS IMMA, exact for {-1,0,+1};  gemm:fp16 = HMMA
CUSTOM_VARIANTS="${CUSTOM_VARIANTS:-auto bit csr-row csr-block csr-cluster block cluster global-sync gemm gemm:fp32:notf32 gemm:int8 block:fp16 cluster:fp16 global-sync:fp16 gemm:fp16}"
# Per-process wall-clock limit (one variant x precision, or one public dtype).
# On timeout / OOM / does-not-fit the entry is logged in run_status.csv, the
# rows it finished are kept, and the next entry runs.
VARIANT_TIMEOUT_S="${VARIANT_TIMEOUT_S:-1800}"
# Public-baseline schedule: matched (dt, pump of dsb-gpu) | library (SB 2.0.0
# defaults); "matched library" runs both.  See python/sb_schedule.py.
SB_SCHEDULES="${SB_SCHEDULES:-matched}"
DT="${DT:-1.0}"   # dSB step size for dsb-gpu and the matched baseline
RESULT_ROOT="${RESULT_ROOT:-results/run_K2000_result}"

RUNNER="scripts/run_gh200_gset_comparison.sh"
INPUT_FILE="${K2000_FILE:-data/K2000/data/WK2000_1.rud}"
TARGET_CUT=33337
STEPS_LIST=(${STEPS_LIST:-200 400 800 1600 3200})

if [[ ! -x "$RUNNER" ]]; then
  echo "Error: executable runner not found: $RUNNER" >&2
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: K2000 input file not found: $INPUT_FILE" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT"
# Append one step directory's run_status.csv to $STATUS_ALL.
collect_status() {
  local f="$1/run_status.csv"
  [[ -f "$f" ]] || return 0
  if [[ ! -s "$STATUS_ALL" ]]; then head -n 1 "$f" > "$STATUS_ALL"; fi
  tail -n +2 "$f" >> "$STATUS_ALL"
}
SUMMARY_FILE="$RESULT_ROOT/run_settings.tsv"
STATUS_ALL="$RESULT_ROOT/run_status_all.csv"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  printf 'instance\ttarget_cut\tbatch\tsteps\tstatus\toutput_dir\n' > "$SUMMARY_FILE"
fi

for steps in "${STEPS_LIST[@]}"; do
  out_dir="$RESULT_ROOT/K2000_${BATCH}_${steps}"
  done_file="$out_dir/.done"
  mkdir -p "$out_dir"

  if [[ -f "$done_file" ]]; then
    echo "Skip completed: K2000, batch=${BATCH}, steps=${steps}"
    continue
  fi

  echo "Run: K2000, target=${TARGET_CUT}, batch=${BATCH}, steps=${steps}"
  printf '%s\t%s\t%s\t%s\tstarted\t%s\n' \
    K2000 "$TARGET_CUT" "$BATCH" "$steps" "$out_dir" >> "$SUMMARY_FILE"

  if GPU_ID="$GPU_ID" \
     REPEATS="$REPEATS" \
     WARMUP="$WARMUP" \
     BATCH="$BATCH" \
     STEPS="$steps" \
     PRECISION="$PRECISION" \
     FASTHARE_ALPHA="$FASTHARE_ALPHA" \
     REDUCTION_MODES="$REDUCTION_MODES" \
     CUSTOM_VARIANTS="$CUSTOM_VARIANTS" \
     VARIANT_TIMEOUT_S="$VARIANT_TIMEOUT_S" \
     SB_SCHEDULES="$SB_SCHEDULES" \
     DT="$DT" \
     OUT_DIR="$out_dir" \
     "$RUNNER" "$INPUT_FILE" "$TARGET_CUT" 2>&1 | tee "$out_dir/run.log"; then
    touch "$done_file"
    collect_status "$out_dir"
    printf '%s\t%s\t%s\t%s\tcompleted\t%s\n' \
      K2000 "$TARGET_CUT" "$BATCH" "$steps" "$out_dir" >> "$SUMMARY_FILE"
  else
    printf '%s\t%s\t%s\t%s\tfailed\t%s\n' \
      K2000 "$TARGET_CUT" "$BATCH" "$steps" "$out_dir" >> "$SUMMARY_FILE"
    echo "Failed: K2000, batch=${BATCH}, steps=${steps} (continuing)" >&2
    collect_status "$out_dir"
  fi
done

echo "All K2000 runs completed. Results: $RESULT_ROOT"
