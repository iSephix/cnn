#!/usr/bin/env bash
set -euo pipefail
tasks=$(realpath "$1")
work=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
mkdir -p "$work"
bash rc23/bank.sh "$build/detectors-rc23"
node run.js strip "$tasks" "$work/blind"
bash rc21/predict.sh "$work/blind" "$work/baseline.json" "$build"
python3 rc23/bank.py "$work/blind" "$work/bank" "$build/detectors-rc23"
python3 rc23/calibrate.py "$work/blind" "$work/baseline.json" "$work/bank" "$work/model.json" --labels "$tasks"
