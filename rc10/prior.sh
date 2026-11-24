#!/usr/bin/env bash
set -euo pipefail
blind=$1
work=$2
mkdir -p "$work"
node rc2/run.js predict "$blind" "$work/base.json" rc2/results/library.json 512
python3 rc3/predict.py "$blind" "$work/base.json" "$work/base.json.trace.json" "$work/prior.json"
python3 rc4/predict.py "$blind" "$work/prior.json" "$work/rc4.json"
python3 rc5/predict.py "$blind" "$work/rc4.json" "$work/rc5.json"
python3 rc6/run.py "$blind" "$work/rc5.json" "$work/rc6.json" free 0
python3 rc7/predict.py "$blind" "$work/rc6.json" "$work/rc7.json"
python3 rc8/run.py "$blind" "$work/rc7.json" rc8/bank.json "$work/reuse.json"
python3 rc8/expand.py "$blind" "$work/reuse.json" "$work/rc8.json"
python3 rc9/run.py "$blind" "$work/rc8.json" "$work/rc9"
python3 rc9/run.py "$blind" "$work/rc9-both.json" "$work/rc9" geometry
python3 rc9/run.py "$blind" "$work/rc9-geometry.json" "$work/rc9" tiles
