#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc19/predict.sh "$blind" "$tmp/baseline.json" "$build"
python3 rc20/emit.py "$blind" "$tmp/object"
mkdir -p "$(dirname "$output")"
python3 rc18/select.py "$tmp/baseline.json" "$tmp/object" "$output"
