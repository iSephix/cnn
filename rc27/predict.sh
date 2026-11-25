#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
bash rc25/predict.sh "$blind" "$work/baseline.json" "$build"
sources=()
for family in relational local patterns topological constraints layout; do
    python3 rc27/emit.py "$blind" "$work/$family" --family "$family"
    sources+=("$work/$family")
done
mkdir -p "$(dirname "$output")"
python3 rc27/arbitrate.py "$blind" "$work/baseline.json" "$output" "${sources[@]}" --promote
