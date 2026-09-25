#!/usr/bin/env bash
set -Eeuo pipefail

# Run the G-set step sweep from the dsb-gpu repository root.
# Output layout:
#   results/run_Gset_result/G<id>_<batch>_<steps>/run.log   (RESULT_ROOT overrides results/run_Gset_result)
#
# Examples:
#   ./run_Gset.sh              # run the complete standard G-set collection
#   ./run_Gset.sh 67           # run only G67
#   ./run_Gset.sh G67 G81      # G prefix is also accepted
#   BATCH=100 REPEATS=5 ./run_Gset.sh 67
#   REDUCTION_MODES="0 1" ./run_Gset.sh 67   # bring FastHare back

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
RESULT_ROOT="${RESULT_ROOT:-results/run_Gset_result}"

RUNNER="scripts/run_gh200_gset_comparison.sh"
DATA_DIR="data/Gset/data"
BEST_KNOWN_CSV="${BEST_KNOWN_CSV:-data/Gset/best_known.csv}"
STEPS_LIST=(${STEPS_LIST:-200 400 800 1600 3200})
DEFAULT_INSTANCES=(
  {1..67}
  70 72 77 81
)

if [[ ! -x "$RUNNER" ]]; then
  echo "Error: executable runner not found: $RUNNER" >&2
  exit 1
fi

if [[ ! -f "$BEST_KNOWN_CSV" ]]; then
  echo "Error: best-known table not found: $BEST_KNOWN_CSV" >&2
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
  printf 'gset\tbest_known\tbatch\tsteps\tstatus\toutput_dir\n' > "$SUMMARY_FILE"
fi

for requested_id in "${INSTANCES[@]}"; do
  id="${requested_id#G}"
  id="${id#g}"

  if [[ ! "$id" =~ ^[0-9]+$ ]]; then
    echo "Error: invalid G-set instance: $requested_id" >&2
    exit 1
  fi

  input_file="$DATA_DIR/G${id}"
  if [[ ! -f "$input_file" ]]; then
    echo "Error: input file not found: $input_file" >&2
    exit 1
  fi

  best_known="$(awk -F, -v instance="G${id}" '
    NR > 1 && $1 == instance {
      gsub(/\r/, "", $2)
      print $2
      exit
    }
  ' "$BEST_KNOWN_CSV")"

  if [[ ! "$best_known" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error: no valid best-known value for G${id} in $BEST_KNOWN_CSV" >&2
    exit 1
  fi

  for steps in "${STEPS_LIST[@]}"; do
    out_dir="$RESULT_ROOT/G${id}_${BATCH}_${steps}"
    done_file="$out_dir/.done"
    mkdir -p "$out_dir"

    if [[ -f "$done_file" ]]; then
      echo "Skip completed: G${id}, batch=${BATCH}, steps=${steps}"
      continue
    fi

    echo "Run: G${id}, target=${best_known}, batch=${BATCH}, steps=${steps}"
    printf '%s\t%s\t%s\t%s\tstarted\t%s\n' \
      "G${id}" "$best_known" "$BATCH" "$steps" "$out_dir" >> "$SUMMARY_FILE"

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
       "$RUNNER" "$input_file" "$best_known" 2>&1 | tee "$out_dir/run.log"; then
      touch "$done_file"
      collect_status "$out_dir"
      printf '%s\t%s\t%s\t%s\tcompleted\t%s\n' \
        "G${id}" "$best_known" "$BATCH" "$steps" "$out_dir" >> "$SUMMARY_FILE"
    else
      printf '%s\t%s\t%s\t%s\tfailed\t%s\n' \
        "G${id}" "$best_known" "$BATCH" "$steps" "$out_dir" >> "$SUMMARY_FILE"
      echo "Failed: G${id}, batch=${BATCH}, steps=${steps} (continuing)" >&2
      collect_status "$out_dir"
    fi
  done
done

echo "All requested runs completed. Results: $RESULT_ROOT"
