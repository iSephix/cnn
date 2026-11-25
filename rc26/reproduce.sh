#!/usr/bin/env bash
set -euo pipefail
one=$(realpath "$1")
two=$(realpath "$2")
work=$(realpath -m "$3")
cd "$(dirname "$0")/.."
mkdir -p "$work/acquire" "$work/holdout"
python3 - "$one/training" "$work" <<'PY'
from pathlib import Path
import shutil
import sys
tasks = sorted(Path(sys.argv[1]).glob('*.json'))
work = Path(sys.argv[2])
for name, paths in [('holdout', tasks[:100]), ('acquire', tasks[100:])]:
    for path in paths:
        shutil.copyfile(path, work/name/path.name)
PY
python3 rc26/test.py
python3 rc26/search.py train "$work/acquire" "$work/model.json"
python3 - "$work/model.json" rc26/model.json <<'PY'
import json
from pathlib import Path
import sys
assert json.loads(Path(sys.argv[1]).read_text()) == json.loads(Path(sys.argv[2]).read_text())
PY
node run.js strip "$work/holdout" "$work/holdout-blind"
bash rc23/bank.sh "$work/build/detectors-rc23"
for arm in fixed learned forward; do
  options=()
  if [[ "$arm" == learned ]]; then options=(--model "$work/model.json"); fi
  if [[ "$arm" == forward ]]; then options=(--forward-only); fi
  python3 rc26/search.py predict "$work/holdout-blind" "$work/holdout-$arm" "${options[@]}"
  python3 rc26/evaluate.py "$work/holdout" "$work/holdout-$arm" "$work/holdout-$arm.json"
done
python3 rc4/transfer.py "$one" "$two" "$work/supp"
python3 rc10/sample.py "$one" "$two" "$work/fresh"
python3 rc11/remaining.py "$one" "$two" "$work/remaining"
for split in arc1 arc2 supp fresh remaining; do
  case "$split" in
    arc1) tasks="$one/evaluation" ;;
    arc2) tasks="$two/evaluation" ;;
    *) tasks="$work/$split" ;;
  esac
  node run.js strip "$tasks" "$work/$split-blind"
  bash rc25/predict.sh "$work/$split-blind" "$work/$split-baseline.json" "$work/build"
  for arm in fixed learned; do
    options=()
    if [[ "$arm" == learned ]]; then options=(--model "$work/model.json"); fi
    python3 rc26/search.py predict "$work/$split-blind" "$work/$split-$arm" "${options[@]}"
    python3 rc26/evaluate.py "$tasks" "$work/$split-$arm" "$work/$split-$arm.json"
    python3 rc26/predict.py "$work/$split-blind" "$work/$split-$arm" \
        "$work/$split-baseline.json" "$work/$split-$arm-pred.json"
    node run.js score "$tasks" "$work/$split-$arm-score.json" "$work/$split-$arm-pred.json"
  done
  python3 rc26/bridge.py "$work/$split-blind" "$work/$split-bridge" \
      "$work/model.json" "$work/build/detectors-rc23"
  python3 rc26/evaluate.py "$tasks" "$work/$split-bridge" "$work/$split-bridge.json"
done
python3 rc26/bridge.py "$work/holdout-blind" "$work/holdout-bridge" \
    "$work/model.json" "$work/build/detectors-rc23"
python3 rc26/evaluate.py "$work/holdout" "$work/holdout-bridge" "$work/holdout-bridge.json"
