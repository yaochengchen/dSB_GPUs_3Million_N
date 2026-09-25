#!/usr/bin/env bash
# Dense +-1 bit path (batch 1) on 1-2 GH200s, coupling matrix split HBM + Grace.
#   GPUS=2 N_LIST="1000000 2000000 3000000" scripts/run_bit_multi.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

GPUS=${GPUS:-2}
N_LIST=${N_LIST:-"500000 1000000 1500000 2000000 2500000 3000000"}
STEPS=${STEPS:-50}
REPEATS=${REPEATS:-3}
PLACEMENT=${PLACEMENT:-auto}          # auto (HBM first, rest Grace) | hbm | grace
HBM_FRACTION=${HBM_FRACTION:-0.85}
TIMEOUT_S=${TIMEOUT_S:-3600}
OUT=${OUT:-results/bit_multi_$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$OUT"
make -j bit_dense_multi
{
  date -u
  echo "gpus=$GPUS n_list=$N_LIST steps=$STEPS repeats=$REPEATS placement=$PLACEMENT hbm_fraction=$HBM_FRACTION"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true
  nvidia-smi topo -m 2>/dev/null || true
  numactl --hardware 2>/dev/null || true
} > "$OUT/environment.txt"

./bit_dense_multi --csv-header > "$OUT/raw.csv"
for n in $N_LIST; do
  echo "=== $(date -u) n=$n"
  timeout "$TIMEOUT_S" ./bit_dense_multi --n "$n" --gpus "$GPUS" --steps "$STEPS" --repeats "$REPEATS" \
      --placement "$PLACEMENT" --hbm-fraction "$HBM_FRACTION" --csv \
      >> "$OUT/raw.csv" 2> "$OUT/n${n}.log" \
    || echo "dsb-gpu-bitpm1,$n,$GPUS,0,1,$STEPS,$PLACEMENT,,,,,,,,,error-exit-$?" >> "$OUT/raw.csv"
  tail -n 3 "$OUT/n${n}.log"
done
echo "results in $OUT/raw.csv"
