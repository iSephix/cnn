#!/usr/bin/env bash
# Supply the ARC-AGI-1 and ARC-AGI-2 data directories from RC11's pinned releases.
set -euo pipefail
one=$(realpath "$1"); two=$(realpath "$2"); work=$(realpath -m "$3")
cd "$(dirname "$0")/.."
mkdir -p "$work"
bash rc12/build.sh "$work/build"
python3 rc12/test.py "$work/build/search"
python3 rc4/transfer.py "$one" "$two" "$work/supp"
python3 rc10/sample.py "$one" "$two" "$work/fresh"
python3 rc11/remaining.py "$one" "$two" "$work/remaining"
for split in arc1 arc2 supp fresh remaining; do
  case "$split" in
    arc1) tasks="$one/evaluation" ;;
    arc2) tasks="$two/evaluation" ;;
    *) tasks="$work/$split" ;;
  esac
  run="$work/run-$split"
  mkdir -p "$run"
  node run.js strip "$tasks" "$run/blind"
  bash rc17/predict.sh "$run/blind" "$run/pred.json" "$work/build"
  node run.js score "$tasks" "$run/score.json" "$run/pred.json"
  cmp "$run/score.json" "rc17/results/$split-score.json"
done
