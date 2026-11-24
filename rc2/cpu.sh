#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
d=${1:?Path to ARC-AGI-1 evaluation directory}
out=${2:-rc2/results}
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
node run.js strip "$d" "$tmpdir/blind"
for rep in 1 2 3; do
  modes=(fresh reuse)
  if [[ $rep == 2 ]]; then modes=(reuse fresh); fi
  for mode in "${modes[@]}"; do
    lib=none
    if [[ $mode == reuse ]]; then lib=rc2/results/library.json; fi
    base="$out/cpu-$mode-$rep"
    node rc2/run.js predict "$tmpdir/blind" "$base.json" "$lib" 2100 10000
    node run.js score "$d" "$base-score.json" "$base.json"
  done
done
