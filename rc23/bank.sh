#!/usr/bin/env bash
set -euo pipefail
dest=$(realpath -m "$1")
revision=e151937e34c8b34f953833a0dab75797fc737ba4
if [[ ! -d "$dest/.git" ]]; then
    git clone --quiet https://github.com/tanmaybisen31/arc-agi-solver.git "$dest"
fi
git -C "$dest" checkout --quiet --detach "$revision"
