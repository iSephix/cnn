#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc10/predict.sh "$blind" "$tmp/prior.json"
bash rc11/pipeline.sh "$blind" "$tmp/prior.json" "$tmp/run"
mkdir -p "$(dirname "$output")"
cp "$tmp/run-rewrite.json" "$output"
cp "$tmp/run-rewrite.json.trace.json" "$output.trace.json"
