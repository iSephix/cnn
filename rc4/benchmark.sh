#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
one=${1:?Pinned ARC-AGI-1 data directory}
two=${2:?Pinned ARC-AGI-2 data directory}
out=${3:?Result directory}
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc3/benchmark.sh "$one" "$two" "$tmp/prior"
for n in 1 2; do
  d="$one/evaluation"
  if [[ $n == 2 ]]; then d="$two/evaluation"; fi
  node run.js strip "$d" "$tmp/blind$n"
  python3 rc4/predict.py "$tmp/blind$n" "$tmp/prior/arc$n.json" "$out/arc$n.json"
  node run.js score "$d" "$out/arc$n-score.json" "$out/arc$n.json"
  python3 rc4/audit.py "$out" "$n" > "$out/arc$n-change.json"
done
