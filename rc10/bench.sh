#!/usr/bin/env bash
# From the repository root. Prediction takes stripped tasks and RC9 output only.
set -euo pipefail
blind=$1
prior=$2
labels=$3
prefix=$4
mkdir -p "$(dirname "$prefix")"
bash rc10/families.sh "$blind" "$prior" "$prefix"
for family in catalog canvas stamps induce resample pointers; do
  node run.js score "$labels" "$prefix-$family-score.json" "$prefix-$family.json"
done
node run.js score "$labels" "$prefix-score.json" "$prefix.json"
