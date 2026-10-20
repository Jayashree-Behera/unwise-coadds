#!/bin/bash
#SBATCH -J coadd_test
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -a 0-1
#SBATCH -o /global/homes/s/shreeb/WISE/Year1/background_correction/logs/test_%A_%a.out
#SBATCH -A desi

# Parameterised copy of the production script. Structure, flags and output
# layout are unchanged; only the leaf of each output directory is renamed per
# test, so results from different variants never overwrite each other.
#
#   sbatch --export=ALL,TAG=baseline,CODEPATH=...,THREAD_ARGS="--threads1 16" run_test.sbatch
#
# TILE        tile id to coadd
# TAG         leaf directory name for this variant's outputs
# CODEPATH    which checkout to run
# THREAD_ARGS thread flags to pass through
# TIMING      value for UNWISE_TIMING (0 = off; baseline has no such support)
# MODE        plain | destripe | both  (default plain)
#
# The default is plain because the destripe run with a fresh EXP_OUTPATH has
# no cached corrected exposures to reuse and would recompute every background
# model. The plain path already exercises everything that was changed: round 1,
# warp recovery, HiLo, round 2 and patch_image.

set -euo pipefail

TILES="${TILE:-1869p121}"
BANDS="1 2"
FORCE=1
FORCE_BG=0
# No colon: an explicitly empty THREAD_ARGS means "pass no thread
# flags", which is how the automatic sizing is exercised.
THREAD_ARGS="${THREAD_ARGS---threads1 16}"
TIMING="${TIMING:-0}"
MODE="${MODE:-plain}"
TAG="${TAG:?set TAG}"
CODEPATH="${CODEPATH:?set CODEPATH}"

META_DIR="/global/cfs/cdirs/desi/users/shreeb/WISE/metadata/neo11/"
MODEL_DIR="/global/cfs/cdirs/cosmo/data/unwise/neo8/unwise-catalog/mod/"
ATLAS="/pscratch/sd/s/shreeb/shreeb/unwise_psf/py/unwise_psf/data/astrom-atlas.fits"
RELEASE="neo11"

# Same parent locations as production; only the leaf changes.
EXP_OUTPATH="/pscratch/sd/s/shreeb/shreeb/WISE/corrected_exposures_formatted/test_${TAG}"
OUTPATH="/pscratch/sd/s/shreeb/shreeb/WISE/corrected_tiles/test_${TAG}"
PLAIN_OUTPATH="/pscratch/sd/s/shreeb/shreeb/WISE/original_tiles/test_${TAG}"

read -ra T <<< "$TILES"
read -ra B <<< "$BANDS"
NPAIRS=$(( ${#T[@]} * ${#B[@]} ))
i=$SLURM_ARRAY_TASK_ID
if (( i >= NPAIRS )); then echo "task $i has no (tile, band) pair"; exit 0; fi
TILE=${T[$(( i / ${#B[@]} ))]}
BAND=${B[$(( i % ${#B[@]} ))]}

module load python
conda activate wise_cosmo
export CROWDSOURCE_DIR="/pscratch/sd/s/shreeb/shreeb/crowdsource/crowdsource"
export UNWISE_META_DIR="$META_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
mkdir -p "$OUTPATH" "$EXP_OUTPATH" "$PLAIN_OUTPATH"

EXTRA=""
[[ "$FORCE" == 1 ]] && EXTRA="$EXTRA --force"

DESTRIPE_EXTRA=""
if [[ "$FORCE_BG" == 1 ]]; then DESTRIPE_EXTRA="--force-bg"; else DESTRIPE_EXTRA="--destripe-cache"; fi

echo "=== TAG=$TAG task $i on $(hostname): tile=$TILE band=w$BAND"
echo "=== CODEPATH=$CODEPATH"
echo "=== commit: $(git -C "$CODEPATH" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "=== THREAD_ARGS=$THREAD_ARGS UNWISE_TIMING=$TIMING"
date

export DESTRIPE_TIMING=50
export UNWISE_TIMING="$TIMING"

if [[ "$MODE" == "destripe" || "$MODE" == "both" ]]; then
echo "=== DESTRIPE RUN"
python -u "$CODEPATH/unwise_coadd.py" \
    --tile "$TILE" --band "$BAND" \
    --outdir "$OUTPATH" \
    --destripe \
    --destripe-model-dir "$MODEL_DIR" \
    --destripe-release "$RELEASE" \
    --destripe-atlas "$ATLAS" \
    --save-corrected-images "$EXP_OUTPATH" \
    $THREAD_ARGS \
    --recover_warped --hi_lo_rej --no_output_masks $EXTRA $DESTRIPE_EXTRA
fi

if [[ "$MODE" == "plain" || "$MODE" == "both" ]]; then
echo "=== PLAIN RUN"
python -u "$CODEPATH/unwise_coadd.py" \
    --tile "$TILE" --band "$BAND" \
    --outdir "$PLAIN_OUTPATH" \
    $THREAD_ARGS \
    --recover_warped --hi_lo_rej --no_output_masks $EXTRA
fi
date
echo "=== DONE TAG=$TAG band=$BAND"
