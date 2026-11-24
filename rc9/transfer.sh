#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc8/transfer.sh "$1" "$2" "$tmp/scores" "$tmp/base"
bash rc9/bench.sh "$tmp/base/blind" "$tmp/base/rc8.json" "$tmp/base/tasks" "$3/supp"
cmp "$3/supp-tiles-score.json" rc9/results/supp-score.json
