#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
mode=${4:-verified}
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc23/bank.sh "$build/detectors-rc23"
bash rc21/predict.sh "$blind" "$tmp/base.json" "$build"
mkdir -p "$(dirname "$output")"
if [[ "$mode" == single-orientation ]]; then
    python3 rc23/bank.py "$blind" "$tmp/bank" "$build/detectors-rc23" --canonical-only
    python3 rc23/calibrate.py "$blind" "$tmp/base.json" "$tmp/bank" rc23/model.json --out "$output"
    exit
fi
if [[ "$mode" != verified ]]; then
    echo "Unknown prediction mode: $mode" >&2
    exit 2
fi
python3 rc23/bank.py "$blind" "$tmp/bank" "$build/detectors-rc23"
python3 rc24/verify.py "$blind" "$tmp/bank" "$tmp/verified" "$build/detectors-rc23"
python3 rc24/palette.py "$blind" "$tmp/palette" "$build/detectors-rc23"
python3 rc24/rank.py "$blind" "$tmp/base.json" "$tmp/verified" "$tmp/palette" rc24/model.json --out "$output"
