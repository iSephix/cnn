#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc12/build.sh "$build"
bash rc11/predict.sh "$blind" "$tmp/prior.json"
bash rc12/cache.sh "$blind" "$build" "$tmp"
python3 rc13/search.py "$blind" "$build/search" "$tmp"/factor
python3 rc13/layers.py "$blind" "$tmp"/layers
mkdir -p "$(dirname "$output")"
python3 rc14/concepts.py "$blind" "$tmp/concepts"
python3 rc15/search.py "$blind" "$tmp/compose"
python3 rc15/stencils.py "$blind" "$tmp/stamp"
python3 rc17/run.py predict "$blind" rc17/model.json "$tmp/stencil" --family stencil
python3 rc17/run.py predict "$blind" rc17/model.json "$tmp/motion" --family motion
python3 rc17/run.py predict "$blind" rc17/model.json "$tmp/chart" --family chart
python3 rc12/rank.py "$blind" "$tmp/prior.json" "$tmp/programs" rc15/model.json "$tmp"/native-{0,1,2} "$tmp"/{mask,motif,beam,factor,layers,concepts,compose,stamp} --out "$tmp/baseline.json"
python3 rc12/rank.py "$blind" "$tmp/prior.json" "$tmp/programs" rc17/ranker.json "$tmp"/native-{0,1,2} "$tmp"/{mask,motif,beam,factor,layers,concepts,compose,stamp,stencil,motion,chart} --out "$tmp/ranked.json"
python3 rc18/run.py "$blind" "$tmp/relation" --family relation
python3 rc18/run.py "$blind" "$tmp/render" --family render
python3 rc18/run.py "$blind" "$tmp/latent" --family latent --binary "$build/search" --workers 6
python3 rc12/rank.py "$blind" "$tmp/prior.json" "$tmp/programs" rc18/ranker.json "$tmp"/native-{0,1,2} "$tmp"/{mask,motif,beam,factor,layers,concepts,compose,stamp,stencil,motion,chart,relation,render,latent} --out "$tmp/extended.json"
python3 rc17/select.py "$tmp/baseline.json" "$tmp/ranked.json" "$tmp/rc17.json" "$tmp"/{stencil,motion,chart}
python3 rc17/select.py "$tmp/rc17.json" "$tmp/extended.json" "$tmp/selected.json" "$tmp"/{relation,render,latent}
python3 rc18/select.py "$tmp/selected.json" "$tmp/relation" "$output"
