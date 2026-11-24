#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
one=399030444e0ab0cc8b4e199870fb20b863846f34
two=f3283f727488ad98fe575ea6a5ac981e4a188e49
for ref in "fchollet/ARC-AGI:$one" "arcprize/ARC-AGI-2:$two"; do
  curl -fsSL --retry 2 "https://codeload.github.com/${ref%:*}/zip/${ref#*:}" -o "$tmp/data.zip"
  unzip -q "$tmp/data.zip" -d "$tmp"
done
d1="$tmp/ARC-AGI-$one/data"
d2="$tmp/ARC-AGI-2-$two/data"
python3 rc6/test.py
bash rc4/benchmark.sh "$d1" "$d2" "$tmp/rc4"
bash rc5/benchmark.sh "$d1" "$d2" "$tmp/rc5" "$tmp/rc4"
bash rc6/bench.sh "$d1" "$d2" "$tmp/rc4" "$tmp/rc5" "$tmp/rc6"
bash rc7/bench.sh "$d1" "$d2" "$tmp/rc6" "$tmp/rc7"
python3 rc8/memory.py "$d1/training" "$tmp/bank.json"
cmp "$tmp/bank.json" rc8/bank.json
python3 rc8/test.py
bash rc8/bench.sh "$d1" "$d2" "$tmp/rc7" "$tmp/reuse"
bash rc8/finish.sh "$d1" "$d2" "$tmp/reuse" "$tmp/final"
for n in 1 2; do
 cmp "$tmp/final/arc$n-score.json" "rc8/results/arc$n-score.json"
done

python3 rc9/test.py
for n in 1 2; do
  data="$d1"
  if [[ $n == 2 ]]; then data="$d2"; fi
  bash rc9/bench.sh "$tmp/reuse/blind$n" "$tmp/final/arc$n.json" "$data/evaluation" "$tmp/rc9/arc$n"
  cmp "$tmp/rc9/arc$n-tiles-score.json" "rc9/results/arc$n-score.json"
done
