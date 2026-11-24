#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
one=399030444e0ab0cc8b4e199870fb20b863846f34
two=f3283f727488ad98fe575ea6a5ac981e4a188e49
for ref in "fchollet/ARC-AGI:$one" "arcprize/ARC-AGI-2:$two"; do
  curl -fsSL --retry 2 "https://codeload.github.com/${ref%:*}/zip/${ref#*:}" -o "$tmp/data.zip"
  unzip -q "$tmp/data.zip" -d "$tmp"
done
d1="$tmp/ARC-AGI-$one/data"
d2="$tmp/ARC-AGI-2-$two/data"
python3 rc6/test.py
bash rc4/benchmark.sh "$d1" "$d2" "$tmp/rc4"
bash rc5/benchmark.sh "$d1" "$d2" "$tmp/rc5" "$tmp/rc4"
bash rc6/bench.sh "$d1" "$d2" "$tmp/rc4" "$tmp/rc5" "$tmp/rc6"
python3 rc6/report.py "$tmp/rc6" > "$tmp/report.json"
python3 - "$tmp/report.json" <<'PY'
import json,sys
actual=json.load(open(sys.argv[1]))
expected=json.load(open('rc6/results/report.json'))
for n in ('1','2'):
    assert actual[n]['runs']['full']==expected[n]['runs']['full']
    assert actual[n]['gains']==expected[n]['gains']
print('Full solver predictions and per-task gains reproduced.')
PY
