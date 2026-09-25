#!/usr/bin/env bash
# Accuracy check of the dense +-1 bit path on Mattis instances J_ij = xi_i xi_j (hidden random xi).
# The ground states are s = +-xi with energy -n(n-1)/2; every run must reach exactly that energy.
#   GPUS=2 N_LIST="500000 1000000 2000000 2500000 3000000" scripts/run_bit_mattis.sh
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

GPUS=${GPUS:-2}
N_LIST=${N_LIST:-"500000 1000000 1500000 2000000 2500000 3000000"}
SEEDS=${SEEDS:-"42 43 44"}              # one hidden xi per seed
STEPS=${STEPS:-50}
REPEATS=${REPEATS:-1}                   # the run is deterministic per seed; use SEEDS for statistics
PLACEMENT=${PLACEMENT:-auto}            # auto (HBM first, rest Grace) | hbm | grace
HBM_FRACTION=${HBM_FRACTION:-0.85}
TIMEOUT_S=${TIMEOUT_S:-3600}
OUT=${OUT:-results/bit_mattis_$(date -u +%Y%m%dT%H%M%SZ)}

mkdir -p "$OUT"
make -j bit_dense_multi || exit 1
{
  date -u
  echo "instance=mattis gpus=$GPUS n_list=$N_LIST seeds=$SEEDS steps=$STEPS repeats=$REPEATS placement=$PLACEMENT hbm_fraction=$HBM_FRACTION"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv 2>/dev/null || true
  numactl --hardware 2>/dev/null || true
} > "$OUT/environment.txt"

./bit_dense_multi --csv-header > "$OUT/raw.csv"
for n in $N_LIST; do
  for seed in $SEEDS; do
    echo "=== $(date -u) n=$n seed=$seed"
    timeout "$TIMEOUT_S" ./bit_dense_multi --instance mattis --n "$n" --seed "$seed" --gpus "$GPUS" \
        --steps "$STEPS" --repeats "$REPEATS" --placement "$PLACEMENT" --hbm-fraction "$HBM_FRACTION" --csv \
        >> "$OUT/raw.csv" 2> "$OUT/n${n}_s${seed}.log"
    rc=$?
    [ $rc -ne 0 ] && echo "dsb-gpu-bitpm1,$n,$GPUS,0,1,$STEPS,$PLACEMENT,,,,,,,,,error-exit-$rc,mattis,,,$seed" >> "$OUT/raw.csv"
    tail -n 2 "$OUT/n${n}_s${seed}.log"
  done
done

# ---- check: energy must equal -n(n-1)/2 in every completed run --------------------------------
# The verdict is printed and also written to $OUT/check.txt (keep raw.csv as pure CSV).
echo
echo "=== check: $OUT/raw.csv"
awk -F, 'NR == 1 { next }
  {
    n = $2; want = -n * (n - 1) / 2
    if ($16 != "completed") { printf "  n=%-8s seed=%-4s %-20s FAIL (%s)\n", n, $20, "", $16; bad++; next }
    ok = (sprintf("%.0f", $14) == sprintf("%.0f", want))
    printf "  n=%-8s seed=%-4s gpus=%s %-7s %.3f s/step  energy %s  expected %.0f  %s\n",
           n, $20, $3, $7, $12, $14, want, ok ? "PASS" : "FAIL"
    if (ok) good++; else bad++
  }
  END { printf "\n%d passed, %d failed\n", good, bad; exit (bad > 0) }' "$OUT/raw.csv" | tee "$OUT/check.txt"
