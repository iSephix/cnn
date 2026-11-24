#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
one=${1:?ARC-AGI-1 data directory}
two=${2:?ARC-AGI-2 data directory}
out=${3:?Output directory}
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
prior=${4:-$tmp/base}
if [[ $# -lt 4 ]]; then bash rc4/benchmark.sh "$one" "$two" "$prior"; fi
for n in 1 2; do
 d="$one/evaluation"
 if [[ $n == 2 ]]; then d="$two/evaluation"; fi
 node run.js strip "$d" "$tmp/blind$n"
 python3 rc5/predict.py "$tmp/blind$n" "$prior/arc$n.json" "$out/arc$n.json"
 node run.js score "$d" "$out/arc$n-score.json" "$out/arc$n.json"
done
