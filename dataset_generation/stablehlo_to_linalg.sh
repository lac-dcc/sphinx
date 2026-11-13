#!/usr/bin/env bash
# stablehlo -> linalg, for NAS-Bench-101 and NAS-Bench-201.
#
#   ./stablehlo_to_linalg.sh [nb101|nb201]
#
# Reads  $GEN_ROOT/<benchmark>/mlir/stablehlo/*.mlir
# Writes $GEN_ROOT/<benchmark>/mlir/linalg/*.mlir
# STABLEHLO_OPT is the path to stablehlo-opt (default: found on the PATH).
# Files that already exist are skipped; failures are listed in linalg_failures.log.
set -uo pipefail

BENCH=${1:-nb101}
ROOT=${GEN_ROOT:-dataset_build}/$BENCH
OPT=${STABLEHLO_OPT:-stablehlo-opt}
SRC=$ROOT/mlir/stablehlo
DST=$ROOT/mlir/linalg
LOG=$ROOT/linalg_failures.log

JOBS=$(( $(nproc) - 2 ))
[ "$JOBS" -lt 1 ] && JOBS=1

[ -d "$SRC" ] || { echo "ERROR: source directory not found: $SRC" >&2; exit 1; }
mkdir -p "$DST"
: > "$LOG"

total=$(find "$SRC" -name '*.mlir' | wc -l)
echo "$total input files, $JOBS workers"
echo "$SRC -> $DST"

export OPT DST LOG
find "$SRC" -name '*.mlir' -print0 \
  | xargs -0 -P "$JOBS" -I{} bash -c '
      f="$1"
      name=$(basename "$f" .mlir)
      out="$DST/$name.mlir"
      [ -s "$out" ] && exit 0
      if ! "$OPT" "$f" \
            --stablehlo-aggressive-folder \
            --stablehlo-aggressive-simplification \
            --stablehlo-legalize-to-linalg=enable-primitive-ops=true \
            --mlir-elide-elementsattrs-if-larger=50 > "$out" 2>>"$LOG"; then
        echo "FAILED: $name" >> "$LOG"
        rm -f "$out"
      fi
    ' _ {}

echo "wrote $(find "$DST" -name '*.mlir' | wc -l) of $total files, $(grep -c '^FAILED:' "$LOG") failures"
