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
python3 rc17/run.py predict "$work/blind" rc17/model.json "$work/stencil" --family stencil
python3 rc17/run.py predict "$work/blind" rc17/model.json "$work/motion" --family motion
python3 rc17/run.py predict "$work/blind" rc17/model.json "$work/chart" --family chart
python3 rc18/run.py "$work/blind" "$work/relation" --family relation
python3 rc18/run.py "$work/blind" "$work/render" --family render
python3 rc18/run.py "$work/blind" "$work/latent" --family latent --binary "$work/build/search" --workers 6
python3 rc19/run.py "$work/blind" "$work/compose2"
python3 rc19/panels.py "$work/blind" "$work/panel" "$work/build/search"
python3 rc19/run.py "$work/blind" "$work/pack" --family pack
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python3 rc19/rank.py "$work/blind" "$work/prior.json" "$work/programs" rc18/ranker.json "$work/ranker.b85" "$work"/native-{0,1,2} "$work"/{mask,motif,beam,factor,layers,concepts,compose,stamp,stencil,motion,chart,relation,render,latent,compose2,panel,pack} --labels "$tasks"
