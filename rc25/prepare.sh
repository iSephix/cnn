#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
work=$(realpath -m "$2")
build=$(realpath -m "$3")
cd "$(dirname "$0")/.."
mkdir -p "$work"
bash rc24/predict.sh "$blind" "$work/base.json" "$build"
bash rc25/build.sh "$build"
python3 rc25/native.py "$blind" "$build/trace-search" "$work/traces"
python3 rc25/emit.py "$blind" "$work/structure"
python3 rc25/emit.py "$blind" "$work/layout" --family layout
