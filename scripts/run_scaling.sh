#!/usr/bin/env bash
# Generate and benchmark N=10k..100k d=100 bipartite instances with CSR dSB.
set -euo pipefail

PYTHON=${PYTHON:-python3}
GPU_ID=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID
# Target cut is analytic (N*d/2), so TTS is meaningful here; 10 runs keeps the
# N=200k point affordable while resolving p_batch to 0.1.
REPEATS=${REPEATS:-10}
WARMUP=${WARMUP:-1}
BATCH=${BATCH:-200}
STEPS=${STEPS:-800}
BASE_SEED=${BASE_SEED:-12345}
GRAPH_SEED=${GRAPH_SEED:-42}
DEGREE=${DEGREE:-100}
OUT_DIR=${OUT_DIR:-results/scaling_$(date -u +%Y%m%dT%H%M%SZ)}
DATA_DIR=${DATA_DIR:-data/scaling}
SIZES=${SIZES:-"10000 20000 50000 100000"}
# dSB step size; auto = python/suggest_dt.py per graph (about 0.49 for d=100:
# dt=1 is past the stability edge on these graphs).
DT=${DT:-auto}
JOBS=${JOBS:-4}

mkdir -p "$OUT_DIR" "$DATA_DIR"
make -j"$JOBS" solve_gset probe
{
  date -u
  uname -a
  echo "physical GPU_ID=$GPU_ID; visible CUDA device=cuda:0"
  nvidia-smi -i "$GPU_ID" --query-gpu=name,uuid,driver_version,memory.total --format=csv
  nvcc --version
} > "$OUT_DIR/environment.txt" 2>&1
./gpu_probe > "$OUT_DIR/gpu_probe.txt"

RAW="$OUT_DIR/raw.csv"
first=1
for n in $SIZES; do
  instance="$DATA_DIR/bipartite_N${n}_d${DEGREE}_seed${GRAPH_SEED}.txt"
  "$PYTHON" python/generate_bipartite_scaling.py "$instance" \
    --n="$n" --degree="$DEGREE" --seed="$GRAPH_SEED"
  target=$((n * DEGREE / 2))
  one="$OUT_DIR/N${n}.csv"
  dt=$DT
  if [[ "$dt" == auto ]]; then dt=$("$PYTHON" python/suggest_dt.py "$instance"); fi
  echo "N=$n dt=$dt" >> "$OUT_DIR/environment.txt"
  ./solve_gset --csv --variant=csr-row --precision=fp32 --no-reduction \
    "--batch=$BATCH" "--steps=$STEPS" "--dt=$dt" "--seed=$BASE_SEED" \
    "--repeats=$REPEATS" "--warmup=$WARMUP" "--target=$target" \
    "$instance" > "$one"
  if [[ $first == 1 ]]; then
    cp "$one" "$RAW"
    first=0
  else
    tail -n +2 "$one" >> "$RAW"
  fi
done

"$PYTHON" python/summarize_tts.py "$RAW" --time-field=solver_s \
  > "$OUT_DIR/scaling_summary.csv"
echo "results: $OUT_DIR"
echo "summary: $OUT_DIR/scaling_summary.csv"
