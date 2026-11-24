#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
v1=399030444e0ab0cc8b4e199870fb20b863846f34
v2=f3283f727488ad98fe575ea6a5ac981e4a188e49
for ref in "fchollet/ARC-AGI:$v1" "arcprize/ARC-AGI-2:$v2"; do
  curl -fsSL --retry 2 "https://codeload.github.com/${ref%:*}/zip/${ref#*:}" -o "$tmpdir/data.zip"
  unzip -q "$tmpdir/data.zip" -d "$tmpdir"
done
d="$tmpdir/ARC-AGI-$v1/data"
node run.js train "$d/training" "$tmpdir/policy.json"
cmp results/policy.json "$tmpdir/policy.json"
node rc2/test.js "$d/training"
bash rc2/benchmark.sh "$d" "$tmpdir/ARC-AGI-2-$v2/data" "$tmpdir/check"
node rc2/transfer.js "$tmpdir/check/transfer.json"
for file in "$tmpdir/check/"*.json; do cmp "$file" "rc2/results/${file##*/}"; done
echo 'Deterministic predictions, traces, cards and scores reproduce.'
