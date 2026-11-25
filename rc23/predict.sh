#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
mode=${4:-learned}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc23/bank.sh "$build/detectors-rc23"
bash rc21/predict.sh "$blind" "$tmp/base.json" "$build"
python3 rc23/bank.py "$blind" "$tmp/bank" "$build/detectors-rc23"
mkdir -p "$(dirname "$output")"
if [[ "$mode" == learned ]]; then
  python3 rc23/calibrate.py "$blind" "$tmp/base.json" "$tmp/bank" rc23/model.json --out "$output"
else
  python3 rc23/select_attempts.py "$tmp/base.json" "$tmp/bank" "$output" --policy "$mode"
fi
