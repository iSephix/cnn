#!/usr/bin/env bash
set -euo pipefail
tasks=$(realpath "$1")
work=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
mkdir -p "$work"
bash rc23/bank.sh "$build/detectors-rc23"
node run.js strip "$tasks" "$work/blind"
bash rc21/predict.sh "$work/blind" "$work/base.json" "$build"
python3 rc23/bank.py "$work/blind" "$work/bank" "$build/detectors-rc23"
python3 rc24/verify.py "$work/blind" "$work/bank" "$work/verified" "$build/detectors-rc23"
python3 rc24/palette.py "$work/blind" "$work/palette" "$build/detectors-rc23"
python3 rc24/rank.py "$work/blind" "$work/base.json" "$work/verified" "$work/palette" "$work/model.json" --labels "$tasks"
