#!/usr/bin/env bash
set -euo pipefail
blind=$1
prior=$2
prefix=$3
mkdir -p "$(dirname "$prefix")"
files=()
for family in catalog canvas stamps induce resample pointers; do
  pred="$prefix-$family.json"
  python3 rc10/run.py "$blind" "$prior" "$pred" "$family"
  files+=("$pred")
done
python3 rc10/merge.py "$prior" "$prefix.json" "${files[@]}"
