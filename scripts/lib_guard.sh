# Shared helpers for the comparison runners (sourced, not executed).
#
# Every solver invocation (one variant x precision x reduction, or one public
# baseline dtype) runs in its own process under `timeout`.  Whatever happens to
# it -- timeout, CUDA/cuBLAS/PyTorch out-of-memory, host OOM killer, plan does
# not fit, unsupported combination -- is classified, written to
# $STATUS_CSV, and the runner moves on to the next entry.  Rows produced
# before a timeout are kept (stdout is line-buffered), so a slow variant still
# contributes time_per_step_s from the repeats it finished.
#
# Required before sourcing: OUT_DIR.  Before record_status: STATUS_INSTANCE,
# BATCH, STEPS, REPEATS.

VARIANT_TIMEOUT_S=${VARIANT_TIMEOUT_S:-1800}
KILL_AFTER_S=${KILL_AFTER_S:-30}
STATUS_CSV="$OUT_DIR/run_status.csv"
STATUS_HEADER='instance,batch,steps,implementation,entry,variant,precision,reduction,status,exit_code,elapsed_s,rows_written,repeats_requested,timeout_s,detail'
echo "$STATUS_HEADER" > "$STATUS_CSV"

# Line-buffer C/C++ stdout so rows reach the file as they are produced.
LINEBUF=()
if command -v stdbuf >/dev/null 2>&1; then
  LINEBUF=(stdbuf -oL -eL)
fi

# guarded_run OUT_CSV ERR_FILE HEADER cmd...
# Runs cmd with stdout -> OUT_CSV, stderr -> ERR_FILE.  Afterwards OUT_CSV
# contains HEADER plus only complete rows (a row cut off by the kill is
# dropped).  Sets GUARD_STATUS, GUARD_RC, GUARD_ELAPSED, GUARD_ROWS,
# GUARD_DETAIL.  Never returns non-zero.
guarded_run() {
  local out=$1 err=$2 header=$3
  shift 3
  local start end rc
  start=$(date +%s.%N)
  set +e
  timeout --signal=TERM "--kill-after=${KILL_AFTER_S}" "$VARIANT_TIMEOUT_S" \
    ${LINEBUF[@]+"${LINEBUF[@]}"} "$@" > "$out.tmp" 2> "$err"
  rc=$?
  set -e
  end=$(date +%s.%N)
  GUARD_RC=$rc
  GUARD_ELAPSED=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.3f", b-a}')

  # Keep the header and every row with the same number of fields as it.
  # An empty HEADER means "take it from the first line the program printed".
  if [[ -z "$header" ]]; then
    header=$(head -n 1 "$out.tmp" | tr -d '\r' || true)
  fi
  if [[ -z "$header" ]]; then
    : > "$out"
    GUARD_ROWS=0
  else
    local nf
    nf=$(awk -F, '{print NF; exit}' <<< "$header")
    {
      echo "$header"
      awk -F, -v nf="$nf" -v h="$header" '
        { sub(/\r$/, "") }
        $0 == h { next }
        NF == nf { print }
      ' "$out.tmp"
    } > "$out"
    GUARD_ROWS=$(( $(wc -l < "$out") - 1 ))
  fi
  rm -f "$out.tmp"

  local timed_out=0
  if [[ $rc == 124 ]]; then
    timed_out=1
  elif [[ $rc == 137 ]] && awk -v e="$GUARD_ELAPSED" -v t="$VARIANT_TIMEOUT_S" \
         'BEGIN{exit !(e >= t)}'; then
    timed_out=1   # TERM ignored, killed after KILL_AFTER_S
  fi

  if [[ $rc == 0 ]]; then
    GUARD_STATUS=ok
  elif [[ $timed_out == 1 ]]; then
    GUARD_STATUS=timeout
  elif grep -qiE 'out of memory|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED|OutOfMemoryError|bad_alloc|cannot allocate memory' "$err"; then
    GUARD_STATUS=oom
  elif [[ $rc == 137 ]]; then
    GUARD_STATUS=killed      # SIGKILL before the timeout: usually the host OOM killer
  elif grep -qiE 'does not fit|needs [0-9]+ B shared' "$err"; then
    GUARD_STATUS=does-not-fit
  elif grep -qiE 'requires|only implemented|not supported|NOT_SUPPORTED|multiple of' "$err"; then
    GUARD_STATUS=unsupported
  else
    GUARD_STATUS=error
  fi

  # Last non-empty stderr line, made CSV-safe.
  GUARD_DETAIL=$(grep -v '^[[:space:]]*$' "$err" 2>/dev/null | tail -n 1 \
                 | tr -d '\r' | tr ',"' ';'"'" | cut -c1-200 || true)
  return 0
}

# record_status IMPLEMENTATION ENTRY VARIANT PRECISION REDUCTION
record_status() {
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
    "$STATUS_INSTANCE" "$BATCH" "$STEPS" "$1" "$2" "$3" "$4" "$5" \
    "$GUARD_STATUS" "$GUARD_RC" "$GUARD_ELAPSED" "$GUARD_ROWS" "$REPEATS" \
    "$VARIANT_TIMEOUT_S" "$GUARD_DETAIL" >> "$STATUS_CSV"
  if [[ $GUARD_STATUS != ok ]]; then
    echo "warning: $1 $2 reduction=$5 -> $GUARD_STATUS after ${GUARD_ELAPSED}s" \
         "(exit $GUARD_RC, $GUARD_ROWS/$REPEATS rows kept): $GUARD_DETAIL" >&2
  else
    echo "ok: $1 $2 reduction=$5 in ${GUARD_ELAPSED}s ($GUARD_ROWS rows)"
  fi
}

# parse_entry ENTRY DEFAULT_PRECISION
# ENTRY is variant[:precision[:flag]], flag currently `notf32`.
# Sets E_VARIANT, E_PRECISION, E_FLAGS (array), E_LABEL (variant column value)
# and E_TAG (file-name tag).
parse_entry() {
  local entry=$1 default_precision=$2 flag
  IFS=: read -r E_VARIANT E_PRECISION flag <<< "$entry"
  E_PRECISION=${E_PRECISION:-$default_precision}
  E_FLAGS=()
  E_LABEL=""
  E_TAG="${E_VARIANT}-${E_PRECISION}"
  case "$flag" in
    "") ;;
    notf32)
      E_FLAGS=(--no-tf32)
      E_LABEL="${E_VARIANT}-notf32"
      E_TAG="${E_TAG}-notf32"
      ;;
    *) echo "warning: unknown flag '$flag' in entry $entry (ignored)" >&2 ;;
  esac
}

# relabel_variant CSV LABEL: rewrite the `variant` column so that e.g. TF32
# and CUDA-core gemm rows are not merged by summarize_tts.py.
relabel_variant() {
  local csv=$1 label=$2
  [[ -z "$label" ]] && return 0
  awk -F, -v OFS=, -v label="$label" '
    NR == 1 { for (i = 1; i <= NF; ++i) if ($i == "variant") col = i; print; next }
    col { $col = label } { print }
  ' "$csv" > "$csv.relabel" && mv "$csv.relabel" "$csv"
}
