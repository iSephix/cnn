#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
one=${1:?Path to ARC-AGI-1 data directory}
two=${2:?Path to ARC-AGI-2 data directory}
out=${3:-rc2/results}
mkdir -p "$out"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
node rc2/run.js mine "$one/training" "$out/library.json"
for n in 1 2; do
  data="$one/evaluation"
  if [[ $n == 2 ]]; then data="$two/evaluation"; fi
  node run.js strip "$data" "$tmpdir/blind-$n"
  for mode in fresh reuse; do
    lib=none
    if [[ $mode == reuse ]]; then lib="$out/library.json"; fi
    for k in 64 512 full; do
      cap=$k
      if [[ $k == full ]]; then cap=2100; fi
      base="$out/arc$n-$mode-$k"
      node rc2/run.js predict "$tmpdir/blind-$n" "$base.json" "$lib" "$cap"
      node run.js score "$data" "$base-score.json" "$base.json"
    done
  done
done
