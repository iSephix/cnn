#!/usr/bin/env bash
# Predict from demonstrations and query inputs. No evaluation answers required.
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc10/prior.sh "$blind" "$tmp/prior"
bash rc10/families.sh "$blind" "$tmp/prior/rc9-tiles.json" "$tmp/final"
mkdir -p "$(dirname "$output")"
cp "$tmp/final.json" "$output"
cp "$tmp/final.json.trace.json" "$output.trace.json"
