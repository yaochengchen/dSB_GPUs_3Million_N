#!/usr/bin/env bash
# Run dsb-gpu and the public PyTorch dSB package on one G-set instance.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 GSET_INSTANCE [TARGET_CUT]" >&2
  exit 2
fi

INSTANCE=$1
TARGET=${2:-}
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

if [[ ! -f "$INSTANCE" ]]; then
  echo "error: instance not found: $INSTANCE" >&2
  exit 2
fi

# With the recommended layout, an instance at data/Gset/data/G22 finds its
# target in data/Gset/best_known.csv.  An explicit second argument overrides
# the table, which is useful when testing another target quality.
if [[ -z "$TARGET" ]]; then
  instance_dir=$(dirname "$INSTANCE")
  instance_name=$(basename "$INSTANCE")
  BEST_KNOWN_FILE=${BEST_KNOWN_FILE:-$instance_dir/../best_known.csv}
  if [[ ! -f "$BEST_KNOWN_FILE" && -f data/Gset/best_known.csv ]]; then
    BEST_KNOWN_FILE=data/Gset/best_known.csv
  fi
  if [[ -f "$BEST_KNOWN_FILE" ]]; then
    TARGET=$(awk -F, -v key="$instance_name" '
      NR > 1 {
        sub(/\r$/, "", $1)
        sub(/\r$/, "", $2)
        if ($1 == key) { print $2; exit }
      }
    ' "$BEST_KNOWN_FILE")
    if [[ -n "$TARGET" ]]; then
      echo "best-known table: $BEST_KNOWN_FILE"
      echo "target cut: $TARGET"
    else
      echo "warning: $instance_name not found in $BEST_KNOWN_FILE" >&2
    fi
  else
    echo "warning: best-known table not found: $BEST_KNOWN_FILE" >&2
  fi
fi

# REPEATS is the number of independent batched runs; p_batch in summary.csv is
# resolved to 1/REPEATS, so TTS curves need >= 50.
REPEATS=${REPEATS:-50}
WARMUP=${WARMUP:-1}
# 512 is the largest batch at which the `bit` plan for N=2000 (K2000, 16 rows
# per block, ~209 KB dynamic shared memory) still fits the 227 KB opt-in limit
# on GH200/H100 with one co-resident block per SM; int8 gemm also needs a
# multiple of 4.  Instances where `bit` does not fit fall back to csr-row
# under `auto` and show up as does-not-fit for `bit` in run_status.csv.
BATCH=${BATCH:-512}
STEPS=${STEPS:-800}
BASE_SEED=${BASE_SEED:-12345}
# Each entry is `variant`, `variant:precision` or `variant:precision:flag`.
# Entries that cannot run on an instance (bit plan does not fit, csr at fp16,
# int8 on non-ternary J, OOM, timeout...) get a line in run_status.csv and an
# empty per-variant CSV; the next entry runs regardless.
# Every variant x precision the solver implements.  `auto` is not a kernel: it
# picks one of bit / csr-row / gemm per instance and is labelled `auto->X`, so
# its rows duplicate the explicit X rows (it measures the dispatcher).
# gemm:fp32:notf32 is gemm on CUDA cores (--no-tf32), labelled gemm-notf32.
CUSTOM_VARIANTS=${CUSTOM_VARIANTS:-"auto bit csr-row csr-block csr-cluster block cluster global-sync gemm gemm:fp32:notf32 gemm:int8 block:fp16 cluster:fp16 global-sync:fp16 gemm:fp16"}
PRECISION=${PRECISION:-fp32}
if [[ ${REDUCTION+x} ]]; then
  # Backward compatibility with the old single-mode switch.
  REDUCTION_MODES=$REDUCTION
else
  # Default: no FastHare.  Set REDUCTION_MODES="0 1" to bring the ablation back.
  REDUCTION_MODES=${REDUCTION_MODES:-"0"}
fi
FASTHARE_ALPHA=${FASTHARE_ALPHA:-0.2}
USE_FASTHARE=0
for reduction in $REDUCTION_MODES; do
  if [[ "$reduction" == 1 ]]; then USE_FASTHARE=1; fi
done
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
    echo "error: REDUCTION_MODES/REDUCTION accepts only 0 and 1" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR"
source "$(dirname -- "${BASH_SOURCE[0]}")/lib_guard.sh"
STATUS_INSTANCE=$(basename "$INSTANCE")

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
CUSTOM_HEADER='implementation,instance,repeat,seed,n_original,n_before_reduction,n_reduced,edges,agents,steps,precision,variant,storage,reduction,early_stopping,objective,target,success,preprocess_s,setup_s,solver_s,reconstruction_s,evaluation_s,total_s,wall_s,reduction_ratio,time_per_step_s,edge_updates_per_s,nnz_solver,coupling_bytes,gpu_memory_bytes'
echo "$CUSTOM_HEADER" > "$CUSTOM_CSV"
for reduction in $REDUCTION_MODES; do
  for entry in $CUSTOM_VARIANTS; do
    parse_entry "$entry" "$PRECISION"
    variant_csv="$OUT_DIR/dsb_gpu_${E_TAG}_reduction${reduction}.csv"
    command=(./solve_gset --csv "--alpha=$FASTHARE_ALPHA"
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
      "$CUSTOM_HEADER" "${command[@]}"
    relabel_variant "$variant_csv" "$E_LABEL"
    record_status dsb-gpu "$entry" "${E_LABEL:-$E_VARIANT}" "$E_PRECISION" "$reduction"
    tail -n +2 "$variant_csv" >> "$CUSTOM_CSV"
  done
done

# The public package is run at every precision that appears in
# CUSTOM_VARIANTS (fp32 and, with gemm:fp16 in the list, fp16), so each of our
# rows has a public row with the same precision to be compared against.
PUBLIC_DTYPES=${PUBLIC_DTYPES:-}
if [[ -z "$PUBLIC_DTYPES" ]]; then
  PUBLIC_DTYPES="$PRECISION"
  for entry in $CUSTOM_VARIANTS; do
    parse_entry "$entry" "$PRECISION"
    PUBLIC_DTYPES="$PUBLIC_DTYPES $E_PRECISION"
  done
  PUBLIC_DTYPES=$(tr ' ' '\n' <<< "$PUBLIC_DTYPES" | sort -u | tr '\n' ' ')
fi

REDUCTION_FILE="$OUT_DIR/fasthare_reduction.txt"
if [[ $USE_FASTHARE == 1 ]]; then
  ./export_fasthare --format=gset "--alpha=$FASTHARE_ALPHA" \
    "--output=$REDUCTION_FILE" "$INSTANCE"
fi

RAW_CSV="$OUT_DIR/raw.csv"
head -n 1 "$CUSTOM_CSV" > "$RAW_CSV"
tail -n +2 "$CUSTOM_CSV" >> "$RAW_CSV"

for public_precision in $PUBLIC_DTYPES; do
  case "$public_precision" in
    fp16) dtype=float16 ;;
    fp32) dtype=float32 ;;
    fp64) dtype=float64 ;;
    int8) continue ;;   # no public int8 dSB exists; compare int8 rows to the fp32/fp16 public rows
    *) echo "warning: no public dtype for precision $public_precision" >&2; continue ;;
  esac
  for sb_schedule in $SB_SCHEDULES; do
  for reduction in $REDUCTION_MODES; do
    public_tag="public_dsb_${public_precision}_${sb_schedule}_reduction${reduction}"
    public_csv="$OUT_DIR/${public_tag}.csv"
    # No --output: rows go to stdout unbuffered (-u), so a timeout keeps the
    # repeats that finished.
    public_command=("$PYTHON" -u python/benchmark_public_gset.py "$INSTANCE"
                    "--agents=$BATCH" "--steps=$STEPS" "--repeats=$REPEATS"
                    "--warmup=$WARMUP" "--seed=$BASE_SEED"
                    "--dtype=$dtype" --device=cuda
                    "--sb-schedule=$sb_schedule" "--sb-time-step=$DT")
    if [[ "$reduction" == 1 ]]; then
      public_command+=("--reduction-file=$REDUCTION_FILE")
    fi
    if [[ -n "$TARGET" ]]; then
      public_command+=("--target=$TARGET")
    fi
    guarded_run "$public_csv" "$OUT_DIR/${public_tag}.stderr" \
      "" "${public_command[@]}"
    record_status public-dsb "public:$public_precision:$sb_schedule" public "$public_precision" "$reduction"
    if [[ -s "$public_csv" ]]; then
      tail -n +2 "$public_csv" >> "$RAW_CSV"
    fi
  done
  done
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
