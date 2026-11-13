#!/bin/bash
# Runs mlir_to_programl on the sample programs and compares the graphs it writes
# (sample_programs/<dialect>/graphs) with the shipped ones (sample_programs/<dialect>/programl).
# Run from the repository root; set TOOL to the binary if it is not on the PATH.
TOOL=${TOOL:-mlir_to_programl}
fail=0
for d in sample_programs/*/; do
    $TOOL "$d" > /dev/null || { echo "$d: the tool failed"; fail=1; continue; }
    same=0; total=0
    for ref in "$d"programl/*.ProgramGraph.pb; do
        total=$((total + 1))
        cmp -s "$ref" "$d"graphs/"$(basename "$ref")" && same=$((same + 1))
    done
    echo "$d: $same of $total graphs identical to the shipped ones"
    [ "$same" -eq "$total" ] || fail=1
done
exit $fail
