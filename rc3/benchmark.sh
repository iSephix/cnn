#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
one=${1:?ARC-AGI-1 data directory}
two=${2:?ARC-AGI-2 data directory}
out=${3:?Output directory}
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for n in 1 2; do
  d="$one/evaluation"
  if [[ $n == 2 ]]; then d="$two/evaluation"; fi
  node run.js strip "$d" "$tmp/$n"
  base="$tmp/rc2-$n.json"
  node rc2/run.js predict "$tmp/$n" "$base" rc2/results/library.json 512
  python3 rc3/predict.py "$tmp/$n" "$base" "$base.trace.json" "$out/arc$n.json"
  node run.js score "$d" "$out/arc$n-score.json" "$out/arc$n.json"
done
