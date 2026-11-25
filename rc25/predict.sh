#!/usr/bin/env bash
set -euo pipefail
blind=$(realpath "$1")
output=$(realpath -m "$2")
build=$(realpath -m "${3:-${TMPDIR:-/tmp}/concept-engine-rc12-build}")
policy=${4:-consensus}
case "$policy" in
    consensus|verified|reserve|learned) ;;
    *) echo "Unknown prediction policy: $policy" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bash rc25/prepare.sh "$blind" "$tmp" "$build"
mkdir -p "$(dirname "$output")"
python3 rc25/arbitrate.py "$blind" "$tmp/base.json" "$tmp/traces" \
    "$tmp/structure" "$tmp/layout" rc25/model.json --policy "$policy" --out "$output"
