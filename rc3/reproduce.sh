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
python3 rc3/test.py
bash rc3/benchmark.sh "$d1" "$d2" "$tmp/res"
for n in 1 2; do
  d="$d1"
  if [[ $n == 2 ]]; then d="$d2"; fi
  python3 rc3/audit.py "$d/evaluation" "$tmp/res" "$n" > "$tmp/res/arc$n-change.json"
  for kind in score change; do cmp "$tmp/res/arc$n-$kind.json" "rc3/results/arc$n-$kind.json"; done
done
