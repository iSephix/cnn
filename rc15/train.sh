#!/usr/bin/env bash
set -euo pipefail
tasks=$(realpath "$1"); work=$(realpath -m "$2")
cd "$(dirname "$0")/.."
mkdir -p "$work"
bash rc12/build.sh "$work/build"
node run.js strip "$tasks" "$work/blind"
bash rc11/predict.sh "$work/blind" "$work/prior.json"
bash rc12/cache.sh "$work/blind" "$work/build" "$work"
python3 rc13/search.py "$work/blind" "$work/build/search" "$work"/factor
python3 rc13/layers.py "$work/blind" "$work"/layers
python3 rc14/concepts.py "$work/blind" "$work/concepts"
python3 rc15/search.py "$work/blind" "$work/compose"
python3 rc15/stencils.py "$work/blind" "$work/stamp"
python3 rc12/rank.py "$work/blind" "$work/prior.json" "$work/programs" "$work/model.json" "$work"/native-{0,1,2} "$work"/{mask,motif,beam,factor,layers,concepts,compose,stamp} --labels "$tasks"
