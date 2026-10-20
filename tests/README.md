# Validation harness

Runs the optimized branch and the baseline it started from side by side on the
same tile, and checks that the outputs match exactly.

Nothing here touches the existing working copy at
`/pscratch/sd/s/shreeb/shreeb/Dustin/unwise-coadds`. The two checkouts used by
these tests live in their own directories.

## 1. Set up (login node, once)

    bash tests/setup_nersc.sh

Clones `destripe_optimize` into `.../Dustin/unwise-coadds-optimize` and commit
`0dc9524` into `.../Dustin/unwise-coadds-baseline`, then builds the optional C
`patch_image` accelerator. If that build fails the pipeline silently falls back
to the numpy implementation, which is still correct, just slower.

## 2. Run

All three variants use the production script's flags and output layout; only
the leaf of each output directory changes, so nothing overwrites anything else.

    OPT=/pscratch/sd/s/shreeb/shreeb/Dustin/unwise-coadds-optimize
    REF=/pscratch/sd/s/shreeb/shreeb/Dustin/unwise-coadds-baseline

    # before
    sbatch --export=ALL,TAG=baseline,CODEPATH=$REF,THREAD_ARGS="--threads1 16" \
           tests/run_test.sbatch

    # after, same thread count -- this is the correctness comparison
    sbatch --export=ALL,TAG=opt_t16,CODEPATH=$OPT,THREAD_ARGS="--threads1 16",TIMING=50 \
           tests/run_test.sbatch

    # after, adaptive -- this is the speed comparison
    sbatch --export=ALL,TAG=opt_auto,CODEPATH=$OPT,THREAD_ARGS="",TIMING=50 \
           tests/run_test.sbatch

Outputs land in the usual places under a `test_<TAG>` leaf:

    .../WISE/corrected_tiles/test_<TAG>
    .../WISE/corrected_exposures_formatted/test_<TAG>
    .../WISE/original_tiles/test_<TAG>

## 3. Compare

    python tests/compare_outputs.py \
        /pscratch/sd/s/shreeb/shreeb/WISE/corrected_tiles/test_baseline \
        /pscratch/sd/s/shreeb/shreeb/WISE/corrected_tiles/test_opt_t16

    python tests/compare_outputs.py \
        /pscratch/sd/s/shreeb/shreeb/WISE/original_tiles/test_baseline \
        /pscratch/sd/s/shreeb/shreeb/WISE/original_tiles/test_opt_t16

Pixel data must be bit-identical, NaN placement included. Headers are reported
separately: FITS writers stamp creation dates, so comparing the files byte for
byte would fail for reasons unrelated to the numbers. `RESULT: BIT-IDENTICAL`
is the pass condition; the exit status is 0 on pass, 1 on mismatch.

Repeat for `test_opt_auto`, which must also match: changing the worker count
must not change the result.

## 4. Timing

`UNWISE_TIMING=50` reports per-stage wall totals every 50 items. Round 1 runs
in worker processes and reports per pid, so those lines need summing across
pids; round 2 and the accumulation loops report once from the parent.
