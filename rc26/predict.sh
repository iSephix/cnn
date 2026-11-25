#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
bash rc25/predict.sh "$blind" "$work/baseline.json" "$build"
python3 rc26/search.py predict "$blind" "$work/programs" --model rc26/model.json
mkdir -p "$(dirname "$output")"
python3 rc26/predict.py "$blind" "$work/programs" "$work/baseline.json" "$output"
