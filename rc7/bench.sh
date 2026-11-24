#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
one=$1
two=$2
prior=$3
out=$4
mkdir -p "$out"
for n in 1 2; do
 d="$one/evaluation"
 if [[ $n == 2 ]]; then d="$two/evaluation"; fi
 node run.js strip "$d" "$out/blind$n"
 python3 rc7/predict.py "$out/blind$n" "$prior/full$n.json" "$out/arc$n.json"
 node run.js score "$d" "$out/arc$n-score.json" "$out/arc$n.json"
done
