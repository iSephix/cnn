#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
build=$(realpath -m "$1")
if [[ ! -x "$build/search" || ! -f "$build/brute2-limited.cpp" ]]; then
    bash "$root/rc12/build.sh" "$build"
fi
# RC12's sources are pinned by its build; cache this adapter by source content.
signature=$(sha256sum "$root/rc12/search.cpp" "$build/brute2-limited.cpp" "$root/rc25/build.sh")
if [[ -x "$build/trace-search" && -f "$build/trace-search.sha256" && $(<"$build/trace-search.sha256") == "$signature" ]]; then
    exit
fi
tmp=$(mktemp -d "$build/.trace-search.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
python3 - "$root/rc12/search.cpp" "$build/brute2-limited.cpp" "$tmp" <<'PY'
from pathlib import Path
import sys
adapter=Path(sys.argv[1]).read_text()
out=Path(sys.argv[3])
a=adapter.index('  set<vector<int>> seen;')
adapter=adapter[:a]+'''  set<vector<int>> seen;
  for (const auto &candidate : ranked) {
    if (candidate.imgs.size()!=data.size()+1) continue;
    vector<int> serialized;
    bool valid=true;
    for (size_t i=0;i<candidate.imgs.size();++i) {
      auto image=sim.rec(i<data.size()?data[i].first:query,candidate.imgs[i]);
      if (min(image.w,image.h)<1 || max(image.w,image.h)>30) {
        valid=false;
        break;
      }
      serialized.push_back(image.h);
      serialized.push_back(image.w);
      serialized.insert(serialized.end(),image.mask.begin(),image.mask.end());
    }
    if (!valid || !seen.insert(serialized).second) continue;
    cout<<"TRACE "<<setprecision(12)<<candidate.score;
    for (int value : serialized) cout<<" "<<value;
    cout<<"\\n";
    if (seen.size()==18) break;
  }
}
'''
(out/'adapter.cpp').write_text(adapter)
search=Path(sys.argv[2]).read_text()
assert 'if (depth >= 20 && curi >= 180) continue;' in search
(out/'brute.cpp').write_text(search.replace('if (depth >= 20 && curi >= 180) continue;',
                                              'if (depth >= 20 && curi >= 480) continue;'))
PY
flags=(-std=c++17 -O2 -ffunction-sections -fdata-sections -I "$build/upstream/headers")
g++ "${flags[@]}" -c "$tmp/brute.cpp" -o "$tmp/brute.o"
objects=("$tmp/brute.o")
for unit in core_functions image_functions image_functions2 efficient normalize brute_size deduce_op pieces compose2; do
    objects+=("$build/$unit.o")
done
g++ "${flags[@]}" "$tmp/adapter.cpp" "${objects[@]}" -Wl,--gc-sections -o "$tmp/search"
printf '%s\n' "$signature" > "$tmp/sha256"
mv "$tmp/search" "$build/trace-search"
mv "$tmp/sha256" "$build/trace-search.sha256"
