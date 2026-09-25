#!/usr/bin/env bash
set -Eeuo pipefail

# Run the QPLIB step sweep from the dsb-gpu repository root.
# Output layout:
#   results/run_qplib_result/QPLIB_<id>_<batch>_<steps>/run.log   (RESULT_ROOT overrides results/run_qplib_result)
#
# Examples:
#   ./run_qplib.sh              # run all 19 instances
#   ./run_qplib.sh 3850         # run only QPLIB_3850
#   ./run_qplib.sh 3506 3850    # run a selected subset
#   REDUCTION_MODES="0 1" ./run_qplib.sh   # bring FastHare back
#   STEPS_LIST="800" PRECISION=fp16 CUSTOM_VARIANTS="gemm" ./run_qplib.sh

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-1}"
# REPEATS=50: p_batch is resolved to 1/REPEATS, so TTS99 needs >= 50 runs.
REPEATS="${REPEATS:-50}"
WARMUP="${WARMUP:-1}"
PRECISION="${PRECISION:-fp32}"
FASTHARE_ALPHA="${FASTHARE_ALPHA:-0.2}"
# FastHare is off by default (only reduction=0).  REDUCTION_MODES="0 1" restores
# the ablation; export_fasthare and fasthare_ablation.csv only run for mode 1.
REDUCTION_MODES="${REDUCTION_MODES:-0}"
# variant[:precision]; every variant solve_qplib implements, fp32 and fp16.
# csr-* are fp32 only; bit/int8 do not apply to real-valued J.
CUSTOM_VARIANTS="${CUSTOM_VARIANTS:-auto csr-row csr-block csr-cluster block cluster global-sync gemm block:fp16 cluster:fp16 global-sync:fp16 gemm:fp16}"
# Per-process wall-clock limit (one variant x precision, or one public dtype).
# On timeout / OOM / does-not-fit the entry is logged in run_status.csv, the
# rows it finished are kept, and the next entry runs.
VARIANT_TIMEOUT_S="${VARIANT_TIMEOUT_S:-1800}"
# Public-baseline schedule: matched (dt, pump of dsb-gpu) | library (SB 2.0.0
# defaults); "matched library" runs both.  See python/sb_schedule.py.
SB_SCHEDULES="${SB_SCHEDULES:-matched}"
DT="${DT:-1.0}"   # dSB step size for dsb-gpu and the matched baseline
RESULT_ROOT="${RESULT_ROOT:-results/run_qplib_result}"

RUNNER="scripts/run_gh200_comparison.sh"
DATA_DIR="data/qplib/data"
STEPS_LIST=(${STEPS_LIST:-200 400 800 1600 3200})
DEFAULT_INSTANCES=(
  3506 3565 3642 3650 3693 3705 3706 3738 3745
  3822 3832 3838 3850 3852 3877 5721 5725 5755 5875
)

declare -A BATCH_FOR=(
  [3506]=1024
  [3565]=2048
  [3642]=256
  [3650]=256
  [3693]=256
  [3705]=1024
  [3706]=512
  [3738]=1024
  [3745]=2048
  [3822]=512
  [3832]=512
  [3838]=512
  [3850]=256
  [3852]=2048
  [3877]=512
  [5721]=512
  [5725]=1024
  [5755]=1024
  [5875]=1024
)

if [[ ! -x "$RUNNER" ]]; then
  echo "Error: executable runner not found: $RUNNER" >&2
  exit 1
fi

if (( $# > 0 )); then
  INSTANCES=("$@")
else
  INSTANCES=("${DEFAULT_INSTANCES[@]}")
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
  printf 'qplib\tbatch\tsteps\tstatus\toutput_dir\n' > "$SUMMARY_FILE"
fi

for id in "${INSTANCES[@]}"; do
  if [[ -z "${BATCH_FOR[$id]+x}" ]]; then
    echo "Error: no batch setting for QPLIB_${id}" >&2
    exit 1
  fi

  input_file="$DATA_DIR/QPLIB_${id}.qplib"
  if [[ ! -f "$input_file" ]]; then
    echo "Error: input file not found: $input_file" >&2
    exit 1
  fi

  batch="${BATCH_FOR[$id]}"

  for steps in "${STEPS_LIST[@]}"; do
    out_dir="$RESULT_ROOT/QPLIB_${id}_${batch}_${steps}"
    done_file="$out_dir/.done"
    mkdir -p "$out_dir"

    if [[ -f "$done_file" ]]; then
      echo "Skip completed: QPLIB_${id}, batch=${batch}, steps=${steps}"
      continue
    fi

    echo "Run: QPLIB_${id}, batch=${batch}, steps=${steps}"
    printf '%s\t%s\t%s\tstarted\t%s\n' \
      "$id" "$batch" "$steps" "$out_dir" >> "$SUMMARY_FILE"

    if GPU_ID="$GPU_ID" \
       REPEATS="$REPEATS" \
       WARMUP="$WARMUP" \
       BATCH="$batch" \
       STEPS="$steps" \
       PRECISION="$PRECISION" \
       FASTHARE_ALPHA="$FASTHARE_ALPHA" \
       REDUCTION_MODES="$REDUCTION_MODES" \
       CUSTOM_VARIANTS="$CUSTOM_VARIANTS" \
       VARIANT_TIMEOUT_S="$VARIANT_TIMEOUT_S" \
       SB_SCHEDULES="$SB_SCHEDULES" \
       DT="$DT" \
       OUT_DIR="$out_dir" \
       "$RUNNER" "$input_file" 2>&1 | tee "$out_dir/run.log"; then
      touch "$done_file"
      collect_status "$out_dir"
      printf '%s\t%s\t%s\tcompleted\t%s\n' \
        "$id" "$batch" "$steps" "$out_dir" >> "$SUMMARY_FILE"
    else
      printf '%s\t%s\t%s\tfailed\t%s\n' \
        "$id" "$batch" "$steps" "$out_dir" >> "$SUMMARY_FILE"
      echo "Failed: QPLIB_${id}, batch=${batch}, steps=${steps} (continuing)" >&2
      collect_status "$out_dir"
    fi
  done
done

echo "All requested runs completed. Results: $RESULT_ROOT"
