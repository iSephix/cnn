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
python3 rc10/test.py
python3 rc4/transfer.py "$d1" "$d2" "$tmp/supp"
python3 rc10/sample.py "$d1" "$d2" "$tmp/fresh"
for split in arc1 arc2 supp fresh; do
  case "$split" in
    arc1) data="$d1/evaluation" ;;
    arc2) data="$d2/evaluation" ;;
    *) data="$tmp/$split" ;;
  esac
  bash rc10/compare.sh "$data" "$tmp/run-$split"
  cmp "$tmp/run-$split/rc10-score.json" "rc10/results/$split-score.json"
done
