#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
dest=$(realpath -m "$1")
mkdir -p "$dest"
src="$dest/upstream"
rev=9407072659de1270358c2ba34c527785214dd68b
if [[ ! -d "$src/.git" ]]; then git clone -q https://github.com/top-quarks/ARC-solution "$src"; fi
git -C "$src" checkout -q "$rev"
units=(core_functions image_functions image_functions2 efficient normalize brute_size brute2 deduce_op pieces compose2)
flags=(-std=c++17 -O2 -ffunction-sections -fdata-sections -I "$src/headers")
for unit in "${units[@]}"; do
  input="$src/src/$unit.cpp"
  if [[ $unit == brute2 ]]; then
    patch -s -o "$dest/brute2-new.cpp" "$input" "$root/limit.patch"
    if ! cmp -s "$dest/brute2-new.cpp" "$dest/brute2-limited.cpp"; then
      mv "$dest/brute2-new.cpp" "$dest/brute2-limited.cpp"
    fi
    input="$dest/brute2-limited.cpp"
  fi
  if [[ ! -f "$dest/$unit.o" || "$dest/$unit.o" -ot "$input" ]]; then
    g++ "${flags[@]}" -c "$input" -o "$dest/$unit.o" &
  fi
  if (( $(jobs -rp | wc -l) >= 3 )); then wait -n; fi
done
wait
objects=()
for unit in "${units[@]}"; do objects+=("$dest/$unit.o"); done
g++ "${flags[@]}" "$root/search.cpp" "${objects[@]}" -Wl,--gc-sections -o "$dest/search.new"
mv "$dest/search.new" "$dest/search"
