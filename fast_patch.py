"""
Drop-in replacement for astrometry.util.miscutils.patch_image.

The upstream version re-derives the list of pixels still needing patching from
the whole image on every iteration:

    needpatching = np.logical_not(mask)
    if required is not None:
        needpatching *= required
    I = np.flatnonzero(needpatching)

For a 1016x1016 L1b frame that is ~1M element logical_not, an optional ~1M
element multiply and a ~1M element scan, repeated once per iteration. The loop
runs until every patchable pixel has been reached, so a frame with a few large
masked blobs (post-cryo frames routinely have ~160,000 masked pixels) pays that
full-image cost many times over.

The set of pixels needing patching can only shrink, and it can only shrink in
one specific way: at the end of each iteration the function writes

    mask.flat[I] = (pn > 0)

which is the only mutation of `mask`, and `required` never changes. So every
index outside `I` keeps whatever `needpatching` value it had (False, or it
would have been in `I`), and the next iteration's index list is exactly
`I[pn == 0]`. We therefore build the index list once and filter it, which
also lets `iy`/`ix` be carried along instead of re-running unravel_index.

Neighbour lookups are done on flat indices (`I + dy*w + dx`) after the
boundary test, which avoids rebuilding two integer coordinate arrays per
offset.

Everything that affects the result is kept in the original order and dtype:

* `psum` is accumulated in `img.dtype` (float32 for WISE), one offset at a
  time, in `dxdy` order, so the float32 rounding sequence is unchanged.
* the non-finite neighbour test still runs before the accumulation, so a NaN
  neighbour still removes that offset entirely rather than poisoning the sum.
* `img.flat[I]` is written for every index in `I`, including those with
  `pn == 0` (which get 0.0), because that is what turns a NaN at an
  unreachable masked pixel into a finite 0.0 for later iterations.
* the final value is `(psum / np.maximum(pn, 1)).astype(img.dtype)`, i.e. a
  float32/int64 division that numpy performs in float64 and then rounds back
  to float32.

As upstream, this modifies both `img` and `mask` in place and returns True on
success, False if an iteration made no progress.
"""
import numpy as np

__all__ = ['patch_image']

try:
    # Optional C accelerator; build with
    #     python setup_patch.py build_ext --inplace
    # It is only used for the exact argument shapes it reproduces bit-for-bit
    # (2-D float32 image, 2-D bool mask, default 4-neighbour offsets).
    from _patch_image import patch_image_c as _patch_image_c
except ImportError:
    _patch_image_c = None

_DEFAULT_DXDY = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def patch_image(img, mask, dxdy=[(-1, 0), (1, 0), (0, -1), (0, 1)],
                required=None):
    '''
    Patch masked pixels by iteratively averaging non-masked neighboring pixels.

    WARNING: this modifies BOTH the "img" and "mask" arrays!

    mask: True for good pixels
    required: if non-None: True for pixels you want to be patched.
    dxdy: Pixels to average in, relative to pixels to be patched.

    Returns True if patching was successful.
    '''
    assert(img.shape == mask.shape)
    assert(len(img.shape) == 2)

    if (_patch_image_c is not None
            and img.dtype == np.float32
            and mask.dtype == bool
            and (required is None or required.dtype == bool)
            and list(dxdy) == _DEFAULT_DXDY):
        return _patch_image_c(img, mask, required)

    return _patch_image_py(img, mask, dxdy, required)


def _patch_image_py(img, mask, dxdy, required):
    h, w = img.shape

    needpatching = np.logical_not(mask)
    if required is not None:
        needpatching *= required
    I = np.flatnonzero(needpatching)
    del needpatching

    iy, ix = np.unravel_index(I, img.shape)

    # .flat, not .reshape(-1): round 1 passes a non-contiguous view into the
    # full frame, which reshape would silently copy.
    imgflat = img.flat
    maskflat = mask.flat

    Nlast = -1
    while True:
        n = len(I)
        if n == 0:
            break
        if n == Nlast:
            return False
        Nlast = n

        psum = np.zeros(n, img.dtype)
        pn = np.zeros(n, int)

        for dx, dy in dxdy:
            # Which of the pixels being patched have this neighbour in bounds?
            ok = None
            if dx < 0:
                ok = (ix >= (-dx))
            elif dx > 0:
                ok = (ix <= (w - 1 - dx))
            if dy < 0:
                ok = (iy >= (-dy)) if ok is None else (ok & (iy >= (-dy)))
            elif dy > 0:
                ok = (iy <= (h - 1 - dy)) if ok is None else (ok & (iy <= (h - 1 - dy)))
            if ok is None:
                sel = np.arange(n)
            else:
                sel = np.flatnonzero(ok)

            J = I[sel] + (dy * w + dx)
            v = imgflat[J]

            # darn, NaN * False = NaN, not zero.
            finite = np.isfinite(v)
            if not finite.all():
                sel = sel[finite]
                J = J[finite]
                v = v[finite]

            m = maskflat[J]
            psum[sel] += v * m
            pn[sel] += m

        imgflat[I] = (psum / np.maximum(pn, 1)).astype(img.dtype)
        done = pn > 0
        maskflat[I] = done

        if done.any():
            keep = ~done
            I = I[keep]
            iy = iy[keep]
            ix = ix[keep]

    return True
