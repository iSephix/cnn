#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc12/build.sh "$build"
bash rc11/predict.sh "$blind" "$tmp/prior.json"
bash rc12/cache.sh "$blind" "$build" "$tmp"
mkdir -p "$(dirname "$output")"
python3 rc12/rank.py "$blind" "$tmp/prior.json" "$tmp/programs" rc12/model.json "$tmp"/native-{0,1,2} "$tmp"/{mask,motif,beam} --out "$output"
