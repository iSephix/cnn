#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
build=$(realpath -m "$1")
if [[ ! -x "$build/search" ]]; then
    bash "$root/rc12/build.sh" "$build"
fi
tmp=$(mktemp -d "$build/.size-search.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
python3 - "$root/rc12/search.cpp" "$tmp/search.cpp" <<'PY'
from pathlib import Path
import sys
source=Path(sys.argv[1]).read_text().replace('argc!=3','(argc!=3 && argc!=5)')
source=source.replace('  auto sizes=bruteSize(x,train);','''  vector<point> sizes;
  if (argc==5) {
    for (auto &pair : train) sizes.push_back(pair.second.sz);
    sizes.push_back({stoi(argv[3]),stoi(argv[4])});
  } else {
    sizes=bruteSize(x,train);
  }''')
source=source.replace('seen.size()==8','seen.size()==16')
Path(sys.argv[2]).write_text(source)
PY
units=(core_functions image_functions image_functions2 efficient normalize brute_size brute2 deduce_op pieces compose2)
objects=()
for unit in "${units[@]}"; do objects+=("$build/$unit.o"); done
g++ -std=c++17 -O2 -ffunction-sections -fdata-sections -I "$build/upstream/headers" \
    "$tmp/search.cpp" "${objects[@]}" -Wl,--gc-sections -o "$tmp/search"
mv "$tmp/search" "$build/size-search"
