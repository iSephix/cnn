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
python3 rc7/test.py
bash rc7/bench.sh "$d1" "$d2" "$tmp/rc6" "$tmp/rc7"
for n in 1 2; do
 cmp "rc7/results/arc$n-score.json" "$tmp/rc7/arc$n-score.json"
 python3 rc7/audit.py "$tmp/rc6/full$n-score.json" "$tmp/rc7/arc$n-score.json" "$tmp/rc7/arc$n.json.trace.json" > "$tmp/change.json"
 cmp "$tmp/change.json" "rc7/results/arc$n-change.json"
done
