#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
v1=399030444e0ab0cc8b4e199870fb20b863846f34
v2=f3283f727488ad98fe575ea6a5ac981e4a188e49
for target in "fchollet/ARC-AGI:$v1" "arcprize/ARC-AGI-2:$v2"; do
  curl -fsSL --retry 2 "https://codeload.github.com/${target%:*}/zip/${target#*:}" -o "$tmpdir/data.zip"
  unzip -q "$tmpdir/data.zip" -d "$tmpdir"
done
d="$tmpdir/ARC-AGI-$v1/data"
d2="$tmpdir/ARC-AGI-2-$v2/data"
node test.js
node run.js train "$d/training" "$tmpdir/policy.json"
cmp "$tmpdir/policy.json" results/policy.json
node ablate.js "$tmpdir/policy.json" "$tmpdir/global.json"
for dataset in 1 2; do
  evaldir="$d/evaluation"
  if [[ $dataset == 2 ]]; then evaldir="$d2/evaluation"; fi
  blind="$tmpdir/blind-$dataset"
  node run.js strip "$evaldir" "$blind"
  node coverage.js "$evaldir" "$tmpdir/coverage.json"
  cmp "$tmpdir/coverage.json" "results/coverage-$dataset.json"
  jobs=(fixed-12 fixed-51 global-12 learned-12 learned-51)
  if [[ $dataset == 2 ]]; then jobs=(arc2); fi
  for tag in "${jobs[@]}"; do
    budget=${tag#*-}
    policy="$tmpdir/policy.json"
    case "$tag" in fixed-*) policy=fixed;; global-*) policy="$tmpdir/global.json";; arc2) budget=51;; esac
    node run.js predict "$blind" "$tmpdir/pred.json" "$policy" "$budget"
    cmp "$tmpdir/pred.json" "results/$tag.json"
    node run.js score "$evaldir" "$tmpdir/score.json" "$tmpdir/pred.json"
    cmp "$tmpdir/score.json" "results/$tag-score.json"
  done
done
