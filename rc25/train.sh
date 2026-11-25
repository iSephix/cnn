#!/usr/bin/env bash
set -euo pipefail
tasks=$(realpath "$1")
work=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
mkdir -p "$work"
node run.js strip "$tasks" "$work/blind"
bash rc25/prepare.sh "$work/blind" "$work" "$build"
python3 rc25/arbitrate.py "$work/blind" "$work/base.json" "$work/traces" \
    "$work/structure" "$work/layout" "$work/model.json" --labels "$tasks"
