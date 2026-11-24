#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
one=$1
two=$2
base4=$3
base5=$4
out=$5
mkdir -p "$out"
for n in 1 2; do
 d="$one/evaluation"
 if [[ $n == 2 ]]; then d="$two/evaluation"; fi
 node run.js strip "$d" "$out/blind$n"
 for mode in rc5 fixed free full; do
   seconds=0.02
   prior=$base4
   method=$mode
   if [[ $mode == full ]]; then seconds=0; prior=$base5; method=free; fi
   python3 rc6/run.py "$out/blind$n" "$prior/arc$n.json" "$out/$mode$n.json" "$method" "$seconds"
   node run.js score "$d" "$out/$mode$n-score.json" "$out/$mode$n.json"
 done
done
