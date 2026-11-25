#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc20/predict.sh "$blind" "$tmp/base.json" "$build"
for family in markers repair route bind regions place; do
    python3 rc21/emit.py "$family" "$blind" "$tmp/$family"
done
mkdir -p "$(dirname "$output")"
python3 rc21/select.py "$tmp/base.json" "$output" "$tmp"/{markers,repair,route,bind,regions,place}
