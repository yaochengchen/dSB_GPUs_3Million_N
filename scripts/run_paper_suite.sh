#!/usr/bin/env bash
# One entry point for QPLIB, K2000, selected G-set, and dense GH200 scaling.
set -euo pipefail
shopt -s nullglob

ROOT_OUT=${ROOT_OUT:-results/paper_$(date -u +%Y%m%dT%H%M%SZ)}
GPU_ID=${GPU_ID:-1}
export GPU_ID CUDA_VISIBLE_DEVICES=$GPU_ID
GSET_NAMES=${GSET_NAMES:-"G22 G55 G67 G77 G81"}
K2000_FILE=${K2000_FILE:-data/K2000/data/WK2000_1.rud}
RUN_QPLIB=${RUN_QPLIB:-1}
RUN_K2000=${RUN_K2000:-1}
RUN_GSET=${RUN_GSET:-1}
RUN_SCALING=${RUN_SCALING:-1}
mkdir -p "$ROOT_OUT"

make -j"${JOBS:-4}" all probe
if [[ ${SKIP_SELFTEST:-0} != 1 ]]; then
  ./selftest 512 4 50 > "$ROOT_OUT/selftest.txt"
fi
export SKIP_SELFTEST=1

if [[ "$RUN_QPLIB" == 1 ]]; then
  if [[ -n ${QPLIB_FILES:-} ]]; then
    read -r -a qplib_files <<< "$QPLIB_FILES"
  else
    qplib_files=(data/qplib/data/*.qplib)
  fi
  for instance in "${qplib_files[@]}"; do
    [[ -f "$instance" ]] || continue
    name=$(basename "$instance" .qplib)
    OUT_DIR="$ROOT_OUT/qplib_$name" scripts/run_gh200_comparison.sh "$instance"
  done
fi

if [[ "$RUN_K2000" == 1 ]]; then
  if [[ -f "$K2000_FILE" ]]; then
    OUT_DIR="$ROOT_OUT/K2000" \
      scripts/run_gh200_gset_comparison.sh "$K2000_FILE" 33337
  else
    echo "warning: K2000 file missing: $K2000_FILE" >&2
  fi
fi

if [[ "$RUN_GSET" == 1 ]]; then
  for name in $GSET_NAMES; do
    instance="data/Gset/data/$name"
    if [[ -f "$instance" ]]; then
      OUT_DIR="$ROOT_OUT/$name" scripts/run_gh200_gset_comparison.sh "$instance"
    else
      echo "warning: G-set file missing: $instance" >&2
    fi
  done
fi

if [[ "$RUN_SCALING" == 1 ]]; then
  OUT="$ROOT_OUT/dense_scaling" scripts/run_dense_scaling.sh
fi

all_summary="$ROOT_OUT/all_quality_summary.csv"
first=1
for summary in "$ROOT_OUT"/qplib_*/summary.csv "$ROOT_OUT"/K2000/summary.csv \
               "$ROOT_OUT"/G*/summary.csv; do
  [[ -f "$summary" ]] || continue
  if [[ $first == 1 ]]; then
    cp "$summary" "$all_summary"
    first=0
  else
    tail -n +2 "$summary" >> "$all_summary"
  fi
done

all_fasthare="$ROOT_OUT/all_fasthare_ablation.csv"
first=1
for ablation in "$ROOT_OUT"/qplib_*/fasthare_ablation.csv \
                "$ROOT_OUT"/K2000/fasthare_ablation.csv \
                "$ROOT_OUT"/G*/fasthare_ablation.csv; do
  [[ -f "$ablation" ]] || continue
  if [[ $first == 1 ]]; then
    cp "$ablation" "$all_fasthare"
    first=0
  else
    tail -n +2 "$ablation" >> "$all_fasthare"
  fi
done

echo "paper results: $ROOT_OUT"
[[ -f "$all_summary" ]] && echo "combined summary: $all_summary"
[[ -f "$all_fasthare" ]] && echo "FastHare ablation: $all_fasthare"
[[ -f "$ROOT_OUT/dense_scaling/summary.csv" ]] && \
  echo "dense scaling summary: $ROOT_OUT/dense_scaling/summary.csv"
