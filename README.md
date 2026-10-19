# unwise-coadds
Unofficial coadds of the WISE mid-infrared satellite imaging

Described in http://arxiv.org/abs/1405.0308 (source at https://github.com/dstndstn/unwise-paper)
and results available from http://unwise.me


## Optional C accelerator

`patch_image` (interpolation over masked pixels) is the single most expensive
pure-numpy step in round 1 and it runs once per L1b frame. `fast_patch.py`
contains a bit-identical numpy rewrite that is about 2.5x faster than
`astrometry.util.miscutils.patch_image`, plus an optional C implementation that
is about 22x faster:

    python setup_patch.py build_ext --inplace

Building is optional. If `_patch_image` is not importable, `fast_patch.py`
silently uses its numpy implementation, and the output is identical either way.

## Timing instrumentation

Setting `UNWISE_TIMING=<N>` turns on per-stage wall-clock accounting for the
main pipeline (`pipeline_timing.py`), with a progress report every `N`
processed frames and a summary per phase. It is completely inert when unset.
This is the no-destripe analogue of `DESTRIPE_TIMING`.
