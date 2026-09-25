#!/usr/bin/env bash
# Run dsb-gpu and the public PyTorch dSB package on the same QPLIB instance.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 INSTANCE.qplib [TARGET_OBJECTIVE]" >&2
  exit 2
fi

INSTANCE=$1
TARGET=${2:-}
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# QPLIB archives commonly place instances and reference solutions in sibling
# data/ and sol/ directories.  When no explicit target is supplied, read the
# published objective from the matching .sol file, for example:
#   data/qplib/data/QPLIB_3506.qplib
#   data/qplib/sol/QPLIB_3506.sol
if [[ -z "$TARGET" ]]; then
  instance_dir=$(dirname "$INSTANCE")
  instance_name=$(basename "$INSTANCE" .qplib)
  SOL_FILE="$instance_dir/../sol/$instance_name.sol"
  if [[ -f "$SOL_FILE" ]]; then
    TARGET=$(awk '$1 == "objvar" {print $2; exit}' "$SOL_FILE")
    if [[ -z "$TARGET" ]]; then
      echo "error: objvar not found in $SOL_FILE" >&2
      exit 2
    fi
    echo "reference solution: $SOL_FILE"
    echo "target objective: $TARGET"
  else
    echo "warning: reference solution not found: $SOL_FILE" >&2
  fi
fi

# p_batch is resolved to 1/REPEATS; TTS curves need >= 50.
REPEATS=${REPEATS:-50}
WARMUP=${WARMUP:-1}
BATCH=${BATCH:-200}
STEPS=${STEPS:-800}
BASE_SEED=${BASE_SEED:-12345}
# variant[:precision], same syntax as run_gh200_gset_comparison.sh.
# QPLIB is real-valued: bit and int8 do not apply; csr-* are fp32 only.
# solve_qplib has no --no-tf32 switch, so fp32 gemm here is always TF32.
CUSTOM_VARIANTS=${CUSTOM_VARIANTS:-"auto csr-row csr-block csr-cluster block cluster global-sync gemm block:fp16 cluster:fp16 global-sync:fp16 gemm:fp16"}
# Default: no FastHare.  Set REDUCTION_MODES="0 1" to bring the ablation back.
REDUCTION_MODES=${REDUCTION_MODES:-"0"}
FASTHARE_ALPHA=${FASTHARE_ALPHA:-0.2}
PRECISION=${PRECISION:-fp32}
# Integration schedule of the public baseline (python/sb_schedule.py):
#   matched  dt=1 and p_k = k/(steps-1), the curve dsb-gpu runs (default)
#   library  simulated-bifurcation 2.0.0 defaults: dt=0.1, p_k = min(k/1000, 1)
# SB_SCHEDULES="matched library" runs both; rows are labelled
# discrete-matched / discrete.
SB_SCHEDULES=${SB_SCHEDULES:-"matched"}
# Integration step size, passed to dsb-gpu (--dt) and, under the matched
# schedule, to the public baseline (--sb-time-step), so both always agree.
# DT=auto picks it per instance from the stability bound of the dynamics
# (python/suggest_dt.py); dt=1 is unstable on dense G-set such as G1-G10.
DT=${DT:-1.0}
for sb_schedule in $SB_SCHEDULES; do
  if [[ "$sb_schedule" != matched && "$sb_schedule" != library ]]; then
    echo "error: SB_SCHEDULES accepts only matched and library" >&2
    exit 2
  fi
done
PYTHON=${PYTHON:-python3}
if [[ "$DT" == auto ]]; then
  DT=$("$PYTHON" python/suggest_dt.py "$INSTANCE")
  echo "DT=auto -> dt=$DT for $(basename "$INSTANCE")"
fi
OUT_DIR=${OUT_DIR:-results/$(date -u +%Y%m%dT%H%M%SZ)}
JOBS=${JOBS:-4}

for reduction in $REDUCTION_MODES; do
  if [[ "$reduction" != 0 && "$reduction" != 1 ]]; then
    echo "error: REDUCTION_MODES accepts only 0 and 1" >&2
    exit 2
  fi
done
USE_FASTHARE=0
RUN_PLAIN=0
for reduction in $REDUCTION_MODES; do
  if [[ "$reduction" == 1 ]]; then USE_FASTHARE=1; fi
  if [[ "$reduction" == 0 ]]; then RUN_PLAIN=1; fi
done

mkdir -p "$OUT_DIR"
source "$(dirname -- "${BASH_SOURCE[0]}")/lib_guard.sh"
STATUS_INSTANCE=$(basename "$INSTANCE" .qplib)

{
  date -u
  uname -a
  echo "physical GPU_ID=$GPU_ID; visible CUDA device=cuda:0"
  echo "batch=$BATCH steps=$STEPS repeats=$REPEATS warmup=$WARMUP"
  echo "precision=$PRECISION FastHare_alpha=$FASTHARE_ALPHA"
  echo "reduction_modes=$REDUCTION_MODES variants=$CUSTOM_VARIANTS"
  echo "variant_timeout_s=$VARIANT_TIMEOUT_S"
  echo "public_sb_schedules=$SB_SCHEDULES dt=$DT"
  nvidia-smi -i "$GPU_ID" --query-gpu=name,uuid,driver_version,memory.total --format=csv
  nvcc --version
  "$PYTHON" --version
  "$PYTHON" -c 'import torch, simulated_bifurcation as sb; print("torch", torch.__version__, "cuda", torch.version.cuda); print("simulated-bifurcation", sb.__version__)'
} > "$OUT_DIR/environment.txt" 2>&1

make -j"$JOBS" all probe
./gpu_probe > "$OUT_DIR/gpu_probe.txt"
if [[ ${SKIP_SELFTEST:-0} != 1 ]]; then
  ./selftest 512 4 50 > "$OUT_DIR/selftest.txt"
fi

CUSTOM_CSV="$OUT_DIR/dsb_gpu.csv"
: > "$CUSTOM_CSV"
for reduction in $REDUCTION_MODES; do
  for entry in $CUSTOM_VARIANTS; do
    parse_entry "$entry" "$PRECISION"
    variant_csv="$OUT_DIR/dsb_gpu_${E_TAG}_reduction${reduction}.csv"
    command=(./solve_qplib --csv "--alpha=$FASTHARE_ALPHA"
             "--batch=$BATCH" "--steps=$STEPS" "--dt=$DT"
             "--precision=$E_PRECISION" "--variant=$E_VARIANT"
             "--seed=$BASE_SEED" "--repeats=$REPEATS" "--warmup=$WARMUP"
             ${E_FLAGS[@]+"${E_FLAGS[@]}"})
    if [[ "$reduction" == 0 ]]; then
      command+=(--no-reduction)
    fi
    if [[ -n "$TARGET" ]]; then
      command+=("--target=$TARGET")
    fi
    command+=("$INSTANCE")
    guarded_run "$variant_csv" "$OUT_DIR/dsb_gpu_${E_TAG}_reduction${reduction}.stderr" \
      "" "${command[@]}"
    relabel_variant "$variant_csv" "$E_LABEL"
    record_status dsb-gpu "$entry" "${E_LABEL:-$E_VARIANT}" "$E_PRECISION" "$reduction"
    if [[ -s "$variant_csv" ]]; then
      if [[ ! -s "$CUSTOM_CSV" ]]; then
        head -n 1 "$variant_csv" > "$CUSTOM_CSV"
      fi
      tail -n +2 "$variant_csv" >> "$CUSTOM_CSV"
    fi
  done
done

RAW_CSV="$OUT_DIR/raw.csv"
cp "$CUSTOM_CSV" "$RAW_CSV"
# append_raw CSV: add CSV's rows to raw.csv (with its header if raw is empty,
# i.e. when every dsb-gpu entry failed).
append_raw() {
  [[ -s "$1" ]] || return 0
  if [[ -s "$RAW_CSV" ]]; then tail -n +2 "$1" >> "$RAW_CSV"; else cat "$1" > "$RAW_CSV"; fi
}

# benchmark_public_dsb.py only accepts float32/float64, so the fp16 dsb-gpu
# rows are compared against the fp32 public rows (as before).
case "$PRECISION" in
  fp64) PUBLIC_DTYPE=float64 ;;
  *)    PUBLIC_DTYPE=float32 ;;
esac

if [[ $USE_FASTHARE == 1 ]]; then
  REDUCTION_FILE="$OUT_DIR/fasthare_reduction.txt"
  ./export_fasthare --format=qplib "--alpha=$FASTHARE_ALPHA" \
    "--output=$REDUCTION_FILE" "$INSTANCE"
fi

for sb_schedule in $SB_SCHEDULES; do
  if [[ $RUN_PLAIN == 1 ]]; then
    PUBLIC_TAG="public_dsb_${sb_schedule}_reduction0"
    PUBLIC_RAW_CSV="$OUT_DIR/${PUBLIC_TAG}.csv"
    public_command=("$PYTHON" -u python/benchmark_public_dsb.py "$INSTANCE"
                    "--agents=$BATCH" "--steps=$STEPS" "--repeats=$REPEATS"
                    "--warmup=$WARMUP" "--seed=$BASE_SEED"
                    "--dtype=$PUBLIC_DTYPE" --device=cuda
                    "--sb-schedule=$sb_schedule" "--sb-time-step=$DT")
    if [[ -n "$TARGET" ]]; then
      public_command+=("--target=$TARGET")
    fi
    guarded_run "$PUBLIC_RAW_CSV" "$OUT_DIR/${PUBLIC_TAG}.stderr" "" \
      "${public_command[@]}"
    record_status public-dsb "public:$PUBLIC_DTYPE:$sb_schedule" public "$PUBLIC_DTYPE" 0
    append_raw "$PUBLIC_RAW_CSV"
  fi

  if [[ $USE_FASTHARE == 1 ]]; then
    PUBLIC_TAG="public_dsb_${sb_schedule}_reduction1"
    PUBLIC_REDUCED_CSV="$OUT_DIR/${PUBLIC_TAG}.csv"
    public_reduced_command=("$PYTHON" -u python/benchmark_public_dsb.py "$INSTANCE"
                            "--reduction-file=$REDUCTION_FILE"
                            "--agents=$BATCH" "--steps=$STEPS"
                            "--repeats=$REPEATS" "--warmup=$WARMUP"
                            "--seed=$BASE_SEED" "--dtype=$PUBLIC_DTYPE" --device=cuda
                            "--sb-schedule=$sb_schedule" "--sb-time-step=$DT")
    if [[ -n "$TARGET" ]]; then
      public_reduced_command+=("--target=$TARGET")
    fi
    guarded_run "$PUBLIC_REDUCED_CSV" "$OUT_DIR/${PUBLIC_TAG}.stderr" "" \
      "${public_reduced_command[@]}"
    record_status public-dsb "public:$PUBLIC_DTYPE:$sb_schedule" public "$PUBLIC_DTYPE" 1
    append_raw "$PUBLIC_REDUCED_CSV"
  fi
done

"$PYTHON" python/summarize_tts.py "$RAW_CSV" > "$OUT_DIR/summary.csv"
"$PYTHON" python/summarize_tts.py "$RAW_CSV" --time-field=solver_s \
  > "$OUT_DIR/summary_solver.csv"
"$PYTHON" python/summarize_tts.py "$RAW_CSV" --time-field=wall_s \
  > "$OUT_DIR/summary_wall.csv"
if [[ $USE_FASTHARE == 1 ]]; then
  "$PYTHON" python/summarize_fasthare_ablation.py "$OUT_DIR/summary.csv" \
    > "$OUT_DIR/fasthare_ablation.csv"
fi

echo "results: $OUT_DIR"
echo "summary: $OUT_DIR/summary.csv"
echo "status:  $STATUS_CSV"
if [[ $USE_FASTHARE == 1 ]]; then
  echo "FastHare ablation: $OUT_DIR/fasthare_ablation.csv"
fi
