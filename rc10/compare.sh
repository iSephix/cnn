#!/usr/bin/env bash
# Compare RC9 and RC10 on any folder of labeled ARC tasks.
set -euo pipefail
tasks=$1
work=$2
mkdir -p "$work"
blind="$work/blind"
node run.js strip "$tasks" "$blind"
bash rc10/prior.sh "$blind" "$work"
node run.js score "$tasks" "$work/rc9-tiles-score.json" "$work/rc9-tiles.json"
bash rc10/bench.sh "$blind" "$work/rc9-tiles.json" "$tasks" "$work/rc10"
python3 rc7/audit.py "$work/rc9-tiles-score.json" "$work/rc10-score.json" "$work/rc10.json.trace.json" > "$work/change.json"
