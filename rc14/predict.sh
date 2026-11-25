#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-/tmp/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc13/predict.sh "$blind" "$tmp/prior.json" "$build"
python3 rc14/concepts.py "$blind" "$tmp/concepts"
mkdir -p "$(dirname "$output")"
python3 rc14/merge.py "$tmp/prior.json" "$tmp/concepts" "$output"
