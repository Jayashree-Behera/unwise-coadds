#!/bin/bash
# One-time setup. Clones the optimized branch and the baseline it is measured
# against into their OWN directories, so the existing working copy at
# .../Dustin/unwise-coadds is never touched.
set -euo pipefail

BASE="/pscratch/sd/s/shreeb/shreeb/Dustin"
REPO="https://github.com/Jayashree-Behera/unwise-coadds.git"

OPT="$BASE/unwise-coadds-optimize"
REF="$BASE/unwise-coadds-baseline"

module load python
conda activate wise_cosmo

# Optimized: branch destripe_optimize
if [[ ! -d "$OPT/.git" ]]; then
    git clone "$REPO" "$OPT"
fi
git -C "$OPT" fetch origin
git -C "$OPT" checkout destripe_optimize
git -C "$OPT" pull --ff-only origin destripe_optimize

# Baseline: the commit the optimization started from, so "before" and "after"
# differ only by these changes.
if [[ ! -d "$REF/.git" ]]; then
    git clone "$REPO" "$REF"
fi
git -C "$REF" fetch origin
git -C "$REF" checkout --detach 0dc9524

# Optional C accelerator for patch_image. Falls back to numpy if this fails.
cd "$OPT"
if CC=gcc python setup_patch.py build_ext --inplace >/tmp/build_patch.log 2>&1; then
    python -c "import _patch_image; print('C module: OK')"
else
    echo "C module: BUILD FAILED (numpy fallback will be used)"
    tail -20 /tmp/build_patch.log
fi

echo
echo "optimized: $OPT  ($(git -C "$OPT" rev-parse --short HEAD))"
echo "baseline : $REF  ($(git -C "$REF" rev-parse --short HEAD))"
echo "untouched: $BASE/unwise-coadds"
