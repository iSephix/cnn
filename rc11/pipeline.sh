#!/usr/bin/env bash
set -euo pipefail
blind=$1
prior=$2
prefix=$3
mkdir -p "$(dirname "$prefix")"
for family in compose regions latent exemplar motion rewrite; do
  output="$prefix-$family.json"
  python3 rc11/run.py "$blind" "$prior" "$output" "$family"
  prior="$output"
done
