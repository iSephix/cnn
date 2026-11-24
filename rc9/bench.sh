#!/usr/bin/env bash
# Run from the repository root, using the final RC8 predictions and traces.
set -euo pipefail
blind=$1
prior=$2
tasks=$3
prefix=$4
mkdir -p "$(dirname "$prefix")"
python3 rc9/run.py "$blind" "$prior" "$prefix"
python3 rc9/run.py "$blind" "$prefix-both.json" "$prefix" geometry
python3 rc9/run.py "$blind" "$prefix-geometry.json" "$prefix" tiles
for mode in single joint both geometry tiles; do
  node run.js score "$tasks" "$prefix-$mode-score.json" "$prefix-$mode.json"
done
