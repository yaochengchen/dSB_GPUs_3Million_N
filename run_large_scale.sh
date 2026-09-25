#!/usr/bin/env bash
set -Eeuo pipefail

# Large-scale paper experiment for dsb-gpu.
#
# Part 1: deterministic d-regular bipartite sparse Max-Cut graphs.
#         Their exact optimum is m = N * degree / 2.
# Part 2: deterministic dense random Ising matrices generated directly on
#         the GPU by dense_scaling (writing O(N^2) text edges is impractical).
#
# Run from the dsb-gpu repository root:
#   chmod +x run_large_scale.sh
#   ./run_large_scale.sh
#
# Run only one part:
#   MODE=sparse ./run_large_scale.sh
#   MODE=dense  ./run_large_scale.sh

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-all}"
GPU_ID="${GPU_ID:-1}"
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-8}"
RESULT_ROOT="${RESULT_ROOT:-results/large_scale}"

# Sparse exact-solution suite.
SPARSE_SIZES="${SPARSE_SIZES:-10000 20000 50000 100000}"
SPARSE_DEGREE="${SPARSE_DEGREE:-100}"
SPARSE_GRAPH_SEED="${SPARSE_GRAPH_SEED:-42}"
SPARSE_BASE_SEED="${SPARSE_BASE_SEED:-12345}"
SPARSE_BATCH="${SPARSE_BATCH:-200}"
SPARSE_STEPS_LIST="${SPARSE_STEPS_LIST:-200 400 800 1600 3200}"
# dSB step size; auto = python/suggest_dt.py per graph.  For the d=100
# bipartite graphs dt=1 is past the stability edge (~0.73): use auto or 0.5.
SPARSE_DT="${SPARSE_DT:-auto}"
SPARSE_REPEATS="${SPARSE_REPEATS:-20}"
SPARSE_WARMUP="${SPARSE_WARMUP:-1}"
SPARSE_VARIANTS="${SPARSE_VARIANTS:-csr-row csr-block csr-cluster}"

# Dense capacity/throughput suite. Batch=1 is intentional: the N x N matrix
# dominates memory, and this suite measures dense scaling rather than TTS.
DENSE_SIZES="${DENSE_SIZES:-10000 20000 40000 80000 120000 160000}"
DENSE_BATCH="${DENSE_BATCH:-1}"
DENSE_STEPS="${DENSE_STEPS:-50}"
DENSE_REPEATS="${DENSE_REPEATS:-5}"
DENSE_WARMUP_STEPS="${DENSE_WARMUP_STEPS:-1}"
DENSE_SEED="${DENSE_SEED:-42}"
DENSE_VARIANTS="${DENSE_VARIANTS:-block cluster auto}"
DENSE_TIMEOUT_S="${DENSE_TIMEOUT_S:-21600}"
DENSE_PYTHON_BASELINE="${DENSE_PYTHON_BASELINE:-1}"

case "$MODE" in
  all|sparse|dense) ;;
  *) echo "Error: MODE must be all, sparse, or dense" >&2; exit 2 ;;
esac

for required in python/generate_bipartite_scaling.py python/summarize_tts.py; do
  [[ -f "$required" ]] || { echo "Error: missing $required" >&2; exit 1; }
done

mkdir -p "$RESULT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_sparse() {
  local data_dir="data/scaling"
  local out_root="$RESULT_ROOT/sparse"
  mkdir -p "$data_dir" "$out_root"
  make -j"$JOBS" solve_gset probe

  for n in $SPARSE_SIZES; do
    local graph="$data_dir/bipartite_N${n}_d${SPARSE_DEGREE}_seed${SPARSE_GRAPH_SEED}.txt"
    local target=$((n * SPARSE_DEGREE / 2))

    "$PYTHON" python/generate_bipartite_scaling.py "$graph" \
      --n="$n" --degree="$SPARSE_DEGREE" --seed="$SPARSE_GRAPH_SEED"

    for steps in $SPARSE_STEPS_LIST; do
      local run_dir="$out_root/N${n}_d${SPARSE_DEGREE}_b${SPARSE_BATCH}_s${steps}"
      local done_file="$run_dir/.done"
      [[ -f "$done_file" ]] && { echo "Skip sparse N=$n steps=$steps"; continue; }
      mkdir -p "$run_dir"

      local dt="$SPARSE_DT"
      if [[ "$dt" == auto ]]; then dt="$("$PYTHON" python/suggest_dt.py "$graph")"; fi
      local combined="$run_dir/raw.csv"
      local first=1
      for variant in $SPARSE_VARIANTS; do
        local one="$run_dir/${variant}.csv"
        echo "Sparse: N=$n target=$target variant=$variant steps=$steps"
        ./solve_gset --csv --no-reduction \
          "--variant=$variant" --precision=fp32 \
          "--batch=$SPARSE_BATCH" "--steps=$steps" "--dt=$dt" \
          "--seed=$SPARSE_BASE_SEED" "--repeats=$SPARSE_REPEATS" \
          "--warmup=$SPARSE_WARMUP" "--target=$target" \
          "$graph" > "$one"
        if (( first )); then
          cp "$one" "$combined"
          first=0
        else
          tail -n +2 "$one" >> "$combined"
        fi
      done

      "$PYTHON" python/summarize_tts.py "$combined" --time-field=solver_s \
        > "$run_dir/summary_solver.csv"
      "$PYTHON" python/summarize_tts.py "$combined" \
        > "$run_dir/summary_total.csv"
      touch "$done_file"
    done
  done
}

run_dense() {
  local out="$RESULT_ROOT/dense"
  if [[ -f "$out/.done" ]]; then
    echo "Skip completed dense suite: $out"
    return
  fi
  mkdir -p "$out"
  echo "Dense matrices are generated deterministically in memory (seed=$DENSE_SEED)."
  GPU_ID="$GPU_ID" \
  N_LIST="$DENSE_SIZES" \
  BATCH="$DENSE_BATCH" \
  STEPS="$DENSE_STEPS" \
  REPEATS="$DENSE_REPEATS" \
  WARMUP_STEPS="$DENSE_WARMUP_STEPS" \
  SEED="$DENSE_SEED" \
  TIMEOUT_S="$DENSE_TIMEOUT_S" \
  PYTHON_BASELINE="$DENSE_PYTHON_BASELINE" \
  CPP_MODES="$DENSE_VARIANTS" \
  OUT="$out" \
  scripts/run_dense_scaling.sh
  touch "$out/.done"
}

if [[ "$MODE" == all || "$MODE" == sparse ]]; then
  run_sparse
fi
if [[ "$MODE" == all || "$MODE" == dense ]]; then
  run_dense
fi

echo "Large-scale suite completed: $RESULT_ROOT"
