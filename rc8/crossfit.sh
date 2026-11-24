#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
data=$1
out=$2
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python3 - "$data" "$tmp" <<'PY'
from pathlib import Path
import hashlib,sys
data,tmp=map(Path,sys.argv[1:])
tasks=sorted(data.glob('*.json'),key=lambda f:hashlib.sha256(f.stem.encode()).digest())
assert len(tasks)==400
for i,p in enumerate(tasks):
    dest=tmp/('acquire' if i<200 else 'test')
    dest.mkdir(exist_ok=True)
    (dest/p.name).write_bytes(p.read_bytes())
PY
python3 rc8/memory.py "$tmp/acquire" "$out/bank.json"
node run.js strip "$tmp/test" "$tmp/blind"
python3 rc8/probe.py "$tmp/blind" "$out" "$out/bank.json"
for mode in frozen color; do node run.js score "$tmp/test" "$out/$mode-score.json" "$out/$mode.json"; done
