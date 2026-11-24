#!/usr/bin/env bash
# From the repository root. Output labels go only to scoring and auditing.
set -euo pipefail
blind=$1
prior=$2
tasks=$3
prefix=$4
bash rc11/pipeline.sh "$blind" "$prior" "$prefix"
node run.js score "$tasks" "$prefix-baseline-score.json" "$prior"
for family in compose regions latent exemplar motion rewrite; do
  node run.js score "$tasks" "$prefix-$family-score.json" "$prefix-$family.json"
done
python3 rc7/audit.py "$prefix-baseline-score.json" "$prefix-rewrite-score.json" "$prefix-rewrite.json.trace.json" > "$prefix-change.json"
