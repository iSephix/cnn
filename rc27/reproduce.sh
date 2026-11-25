#!/usr/bin/env bash
set -euo pipefail
one=$(realpath "$1")
two=$(realpath "$2")
work=$(realpath -m "$3")
cd "$(dirname "$0")/.."
mkdir -p "$work"
python3 rc27/test.py
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
    bash rc27/predict.sh "$work/$split-blind" "$work/$split.json" "$work/build"
    node run.js score "$tasks" "$work/$split-score.json" "$work/$split.json"
    cmp "$work/$split-score.json" "rc27/results/$split-score.json"
done
