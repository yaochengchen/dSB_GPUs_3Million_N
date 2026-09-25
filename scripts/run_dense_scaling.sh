#!/usr/bin/env bash
# Dense high-entropy GH200 capacity scaling.  Physical GPU 1 is the default.
#
# What changed relative to the first version of this experiment, and why:
#
#   * gemm is in the default variant list.  It is the only dense formulation
#     that parallelises over rows, so at batch=1 it is the only one that can
#     use more than `cluster` SMs.  The earlier run compared block/cluster
#     (1-8 blocks on 132 SMs) against cuBLAS GEMV at the HBM roofline and
#     lost 18-56x for that reason alone.  `auto` now resolves to gemm below
#     64 replicas.
#
#   * A batch sweep (BATCH_LIST) instead of batch=1.  At batch=1 every step is a
#     bandwidth-bound stream of J and nothing beats the roofline; the point of
#     dSB is many replicas, and the per-replica cost falls with batch until the
#     GEMM becomes compute bound.  The interesting figure is time/step vs batch
#     at fixed N.
#
#   * A precision sweep (PRECISION_LIST).  FP16 storage halves the bytes per
#     step and moves the HBM boundary from N ~ 176k to N ~ 250k on 144 GB.
#     The PyTorch baseline is run at the same dtype.
#
#   * N_LIST reaches past the HBM boundary so that `auto` really spills J to
#     Grace memory (grace_bytes > 0).  The earlier N_LIST topped out at 160k,
#     which in FP32 is 102 GB and still fits, so the unified-memory claim was
#     never actually exercised.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

GPU_ID=${GPU_ID:-1}
N_LIST=${N_LIST:-"10000 20000 40000 80000 120000 160000 200000 240000 300000"}
BATCH_LIST=${BATCH_LIST:-${BATCH:-"1 8 64"}}
PRECISION_LIST=${PRECISION_LIST:-"fp32 fp16"}
STEPS=${STEPS:-50}
REPEATS=${REPEATS:-3}
WARMUP_STEPS=${WARMUP_STEPS:-1}
SEED=${SEED:-42}
HBM_FRACTION=${HBM_FRACTION:-0.85}
TIMEOUT_S=${TIMEOUT_S:-3600}
PYTHON_BASELINE=${PYTHON_BASELINE:-1}
CPP_MODES=${CPP_MODES:-"gemm auto"}   # cluster already measured; add it back if needed
GRACE_NUMA_NODE=${GRACE_NUMA_NODE:-}
# auto | hbm | grace.  Empty keeps the old rule (auto only for variant=auto,
# hbm for everything else).  Set MATRIX_MEMORY=auto so a gemm-only sweep can
# also spill J to Grace once it no longer fits the HBM budget.
MATRIX_MEMORY=${MATRIX_MEMORY:-}
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-results/dense_scaling_${STAMP}}

mkdir -p "$OUT"
make -j dense_scaling

export CUDA_VISIBLE_DEVICES="$GPU_ID"
NUMA_PREFIX=()
if [[ -n "$GRACE_NUMA_NODE" ]] && command -v numactl >/dev/null 2>&1; then
  NUMA_PREFIX=(numactl "--cpunodebind=$GRACE_NUMA_NODE" \
                       "--membind=$GRACE_NUMA_NODE")
fi

{
  echo "physical_gpu=$GPU_ID"
  echo "process_cuda_device=0"
  echo "n_list=$N_LIST"
  echo "batch_list=$BATCH_LIST"
  echo "precision_list=$PRECISION_LIST"
  echo "steps=$STEPS"
  echo "repeats=$REPEATS"
  echo "warmup_steps=$WARMUP_STEPS"
  echo "seed=$SEED"
  echo "hbm_fraction=$HBM_FRACTION"
  echo "matrix_memory=${MATRIX_MEMORY:-per-variant}"
  echo "cpp_modes=$CPP_MODES"
  echo "grace_numa_node=${GRACE_NUMA_NODE:-automatic}"
  grep Coherent /proc/driver/nvidia/params 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv 2>/dev/null || true
  numactl --hardware 2>/dev/null || true
} > "$OUT/environment.txt"

HEADER='implementation,n,repeat,seed,batch,steps,precision,requested_variant,selected_variant,requested_memory,selected_memory,cluster,matrix_bytes,hbm_bytes,grace_bytes,hbm_free_before,hbm_total,generation_s,gpu_s,time_per_step_s,dense_interactions_per_s,effective_matrix_GB_s,status,error'
echo "$HEADER" > "$OUT/raw.csv"

append_csv() {
  local temporary=$1
  tail -n +2 "$temporary" >> "$OUT/raw.csv"
}

record_failure() {
  local implementation=$1 n=$2 batch=$3 precision=$4 variant=$5 memory=$6
  local status=$7 message=$8
  echo "$implementation,$n,0,$SEED,$batch,$STEPS,$precision,$variant,,$memory,,0,0,0,0,0,0,0,0,0,0,0,$status,$message" \
    >> "$OUT/raw.csv"
}

for precision in $PRECISION_LIST; do
for batch in $BATCH_LIST; do
for n in $N_LIST; do
  for variant in $CPP_MODES; do
    memory=${MATRIX_MEMORY:-hbm}
    if [[ -z "$MATRIX_MEMORY" && "$variant" == auto ]]; then
      memory=auto
    fi
    tag="${variant}_${precision}_b${batch}_${n}"
    temporary="$OUT/cpp_${tag}.csv"
    set +e
    timeout "$TIMEOUT_S" "${NUMA_PREFIX[@]}" ./dense_scaling \
        --n="$n" --batch="$batch" --steps="$STEPS" \
        --repeats="$REPEATS" --warmup-steps="$WARMUP_STEPS" \
        --seed="$SEED" --variant="$variant" --precision="$precision" \
        --matrix-memory="$memory" --hbm-fraction="$HBM_FRACTION" --csv \
        > "$temporary" 2> "$OUT/cpp_${tag}.stderr"
    status=$?
    set -e
    if [[ $status == 0 ]]; then
      append_csv "$temporary"
    elif [[ $status == 124 ]]; then
      record_failure dsb-gpu-dense "$n" "$batch" "$precision" "$variant" \
        "$memory" timeout "exceeded-${TIMEOUT_S}s"
    else
      record_failure dsb-gpu-dense "$n" "$batch" "$precision" "$variant" \
        "$memory" error "process-exit-$status"
    fi
  done

  if [[ "$PYTHON_BASELINE" == 1 ]]; then
    tag="python_${precision}_b${batch}_${n}"
    temporary="$OUT/${tag}.csv"
    set +e
    timeout "$TIMEOUT_S" "${NUMA_PREFIX[@]}" python3 python/benchmark_dense_scaling.py \
        --n="$n" --batch="$batch" --steps="$STEPS" \
        --repeats="$REPEATS" --warmup-steps="$WARMUP_STEPS" \
        --seed="$SEED" --precision="$precision" > "$temporary" \
        2> "$OUT/${tag}.stderr"
    status=$?
    set -e
    if [[ $status == 0 ]]; then
      append_csv "$temporary"
    elif [[ $status == 124 ]]; then
      record_failure python-pytorch-dense "$n" "$batch" "$precision" python \
        hbm timeout "exceeded-${TIMEOUT_S}s"
    else
      record_failure python-pytorch-dense "$n" "$batch" "$precision" python \
        hbm error "process-exit-$status"
    fi
  fi
done
done
done

python3 python/summarize_dense_scaling.py "$OUT/raw.csv" \
  --summary "$OUT/summary.csv" --plot "$OUT/dense_scaling.png"

echo "raw: $OUT/raw.csv"
echo "summary: $OUT/summary.csv"
echo "plot: $OUT/dense_scaling.png"
