#!/usr/bin/env bash
# Build the candidate pool from stripped ARC tasks. Run from repository root.
set -euo pipefail
blind=$1
build=$2
out=$3
workers=${4:-3}
mkdir -p "$out"
for mode in 0 1 2; do
    python3 rc12/native.py "$blind" "$build/search" "$out/native-$mode" --mode "$mode" --workers "$workers"
done
python3 rc12/pool.py "$blind" "$out/programs" --workers "$workers"
python3 rc12/residual.py "$blind" "$build/search" "$out/mask" --workers "$workers"
python3 rc12/motifs.py "$blind" "$build/search" "$out/motif" --workers "$workers"
python3 rc12/native.py "$blind" "$build/search" "$out/beam" --workers "$workers" --depth 3
