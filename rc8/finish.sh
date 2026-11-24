#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
one=$1
two=$2
base=$3
out=$4
mkdir -p "$out"
for n in 1 2; do
 d="$one/evaluation"
 if [[ $n == 2 ]]; then d="$two/evaluation"; fi
 python3 rc8/expand.py "$base/blind$n" "$base/arc$n.json" "$out/arc$n.json"
 node run.js score "$d" "$out/arc$n-score.json" "$out/arc$n.json"
 python3 rc7/audit.py "rc7/results/arc$n-score.json" "$out/arc$n-score.json" "$out/arc$n.json.trace.json" > "$out/arc$n-change.json"
done
