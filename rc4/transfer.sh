#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
one=${1:?ARC-1 data}
two=${2:?ARC-2 data}
out=${3:?Result directory}
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python3 rc4/transfer.py "$one" "$two" "$tmp/tasks"
node run.js strip "$tmp/tasks" "$tmp/blind"
node rc2/run.js predict "$tmp/blind" "$tmp/base.json" rc2/results/library.json 512
python3 rc3/predict.py "$tmp/blind" "$tmp/base.json" "$tmp/base.json.trace.json" "$tmp/prior.json"
python3 rc4/predict.py "$tmp/blind" "$tmp/prior.json" "$tmp/new.json"
for mode in prior new; do
  node run.js score "$tmp/tasks" "$out/$mode-score.json" "$tmp/$mode.json"
done
