"""
Remove the per-channel ("lane") stripe pattern from WISE W1/W2 L1b frames.

A star model (the crowdsource model image minus its sky, taken from the
unWISE coadd tiles overlapping the frame) is projected onto the frame and
subtracted. The stripe pattern is then estimated from the masked,
sky-subtracted residual and removed from the frame.

This module holds only the correction *algorithm* -- it has no knowledge of
unwise_coadd.py's frame selection, multiprocessing, or file layout. The
single entry point used by unwise_coadd.py --destripe is `destripe_frame()`,
which corrects one frame already loaded in memory (no re-reading from
disk, since unwise_coadd.py's round 1 has already read the int/unc/msk
files for its own purposes).
"""
import os
import sys
import numpy as np
import fitsio

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.convolution import convolve, Box2DKernel

from astrometry.util.resample import resample_with_wcs, OverlapError

from scipy.ndimage import binary_dilation, gaussian_filter1d, gaussian_filter, label
from scipy.fft import next_fast_len, rfft, irfft

from functools import lru_cache

from unwise_utils import get_coadd_tile_wcs, zeropointToScale


# ---------------------------------------------------------------------------
# Stage timing.
#
# --destripe costs little extra CPU but a lot of extra wall time, which points
# at blocking I/O rather than computation. Accumulate per-stage wall time in
# each worker process so a run can say where the time actually goes, instead
# of guessing. Accumulating is a couple of time() calls per stage; printing is
# off unless $DESTRIPE_TIMING is set to a frame interval, so this is inert in
# normal runs.
# ---------------------------------------------------------------------------
import time

_STAGE_TIME = {}
_STAGE_N = {}
_FRAMES_TIMED = 0


class _stage(object):
    """Context manager accumulating wall time under `name`."""
    __slots__ = ('name', 't0')

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.t0
        _STAGE_TIME[self.name] = _STAGE_TIME.get(self.name, 0.0) + dt
        _STAGE_N[self.name] = _STAGE_N.get(self.name, 0) + 1
        return False


def timing_report():
    """One-line summary of accumulated stage times for this process."""
    if not _STAGE_TIME:
        return 'destripe timing: nothing recorded'
    total = sum(_STAGE_TIME.values())
    parts = ['%s %.1fs/%d' % (k, _STAGE_TIME[k], _STAGE_N[k])
             for k in sorted(_STAGE_TIME, key=lambda k: -_STAGE_TIME[k])]
    mi = _load_model_tile.cache_info()
    ki = _load_unwise_mask_tile.cache_info()
    return ('destripe timing pid=%d frames=%d total=%.1fs | %s | '
            'model cache %d/%d hit, mask cache %d/%d hit'
            % (os.getpid(), _FRAMES_TIMED, total, '  '.join(parts),
               mi.hits, mi.hits + mi.misses, ki.hits, ki.hits + ki.misses))


def _maybe_report_timing():
    global _FRAMES_TIMED
    _FRAMES_TIMED += 1
    every = int(os.environ.get('DESTRIPE_TIMING', '0') or 0)
    if every > 0 and (_FRAMES_TIMED % every) == 0:
        print(timing_report())

# crowdsource is not pip-installed; unwise_bgcorr_coadd.py located it by
# appending to sys.path. Do the same, but from $CROWDSOURCE_DIR when set so
# the path is not baked in. Resolved lazily and cached by get_sky_im().
_CROWDSOURCE_DEFAULTS = [
    "/pscratch/sd/s/shreeb/shreeb/crowdsource/crowdsource",
    "/pscratch/sd/s/shreeb/shreeb/crowdsource",
]
_sky_im = None


def get_sky_im():
    """
    Import crowdsource_base.sky_im, extending sys.path if needed.

    Raises ImportError with the paths tried; callers should invoke this once
    up front (see unwise_coadd.get_destripe_context) so a missing crowdsource
    aborts the job rather than silently failing every frame.
    """
    global _sky_im
    if _sky_im is not None:
        return _sky_im
    tried = []
    for cand in ([os.environ["CROWDSOURCE_DIR"]]
                 if os.environ.get("CROWDSOURCE_DIR") else _CROWDSOURCE_DEFAULTS):
        for p in (cand, os.path.join(cand, "crowdsource")):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
            tried.append(p)
    try:
        from crowdsource_base import sky_im
    except ImportError as e:
        raise ImportError(
            "could not import crowdsource_base.sky_im, needed by --destripe; "
            "set $CROWDSOURCE_DIR to your crowdsource checkout. Tried: %s (%s)"
            % (", ".join(tried), e))
    _sky_im = sky_im
    return _sky_im

_TILE_W = 2048
_TILE_PS = 2.75 / 3600.0          # unWISE coadd pixel scale [deg/px]
_EXP_HALFDIAG = 1016 * 2.75 / 3600.0 * np.sqrt(2) / 2   # ~0.549 deg
_OVERLAP_PAD = _EXP_HALFDIAG / _TILE_PS + 20            # ~738 px, +20 px safety for SIP/rounding

# unWISE artifact-mask bit groups
_PSF_BITS = sum(1 << b for b in [0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                 27, 28, 29, 30])
_GALAXY_BIT = (1 << 9)
_BIGOBJ_BIT = (1 << 10)

_PSF_DILATION = 1
_GALAXY_DILATION = 6
_BIGOBJ_DILATION = 6

# Per-exposure summary statistics; see _collect_stats. Written as FITS header
# cards (by save_corrected_exposure) and as unwise_coadd.py WISE-table columns
# (lower-cased) when --destripe is active.
STATS_KEYS = ["BGSTAT", "BGWRANGE", "BGMASKF", "BGSKYMED", "BGSKYRNG",
              "BGLANEAM", "BGCOVF", "BGCHOFF"]
_SENTINEL = -999.0   # FITS headers cannot hold NaN

# Stamped into saved corrected exposures as UNW_DSTV, and required to match
# before one is reused as a cache entry. Bump this whenever a change to the
# correction would make previously saved frames wrong: files written by an
# older version are then ignored and recomputed rather than silently reused.
# 1 = unwise_bgcorr_coadd.py wrapper (recomputed its own badbits/zeropoint,
#     omitting bit 19 'hard-saturated' on cryo-phase frames)
# 2 = in-pipeline --destripe, using round 1's own maskbits and zeropoint
# 3 = adds the optional per-channel offset correction (--destripe-channels)
DESTRIPE_VERSION = 3


def exposure_overlaps_tile(exp_ra, exp_dec, tile_ra, tile_dec):
    """
    Vectorized, conservative test: True if an exposure centred at (exp_ra, exp_dec)
    can overlap the 2048x2048 unWISE tile centred at (tile_ra, tile_dec), for any
    exposure rotation. Gnomonic projection onto the tile's tangent plane, then
    check against the tile box expanded by the exposure half-diagonal.
    Never drops an exposure that touches the tile; keeps a few that don't.
    """
    a = np.radians(np.asarray(exp_ra, dtype=float))
    d = np.radians(np.asarray(exp_dec, dtype=float))
    a0 = np.radians(tile_ra)
    d0 = np.radians(tile_dec)

    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.cos(d) * np.sin(a - a0) / cosc
        eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc

    # unWISE tile WCS: CRPIX = (W+1)/2, CD1_1 = -ps, CD2_2 = +ps
    crpix = (_TILE_W + 1) / 2.0
    x = crpix - np.degrees(xi) / _TILE_PS
    y = crpix + np.degrees(eta) / _TILE_PS

    lo, hi = 0.5 - _OVERLAP_PAD, _TILE_W + 0.5 + _OVERLAP_PAD
    return (cosc > 0) & (x > lo) & (x < hi) & (y > lo) & (y < hi)


_ATLAS_COORD_CACHE = {}


def _atlas_tile_coords(atlas_data):
    """Per-worker cache of the atlas SkyCoord array, instead of rebuilding it per exposure."""
    key = id(atlas_data)
    if key not in _ATLAS_COORD_CACHE:
        _ATLAS_COORD_CACHE.clear()  # only one atlas per worker process lifetime
        _ATLAS_COORD_CACHE[key] = SkyCoord(
            ra=atlas_data['CRVAL'][:, 0] * u.deg, dec=atlas_data['CRVAL'][:, 1] * u.deg)
    return _ATLAS_COORD_CACHE[key]


_ATLAS_RADEC_CACHE = {}


def _atlas_radec_map(atlas_data):
    """
    Per-worker cache of {coadd_id: (ra, dec)} from the atlas table's precise
    CRVAL column. Used to build each tile's WCS analytically via
    unwise_utils.get_coadd_tile_wcs -- the same construction unwise_coadd.py
    itself uses for every tile WCS, including the final coadd's own cowcs --
    instead of trusting it to a model/mask FITS file's header, which could
    in principle be stale or written with different precision.
    """
    key = id(atlas_data)
    if key not in _ATLAS_RADEC_CACHE:
        _ATLAS_RADEC_CACHE.clear()  # only one atlas per worker process lifetime
        _ATLAS_RADEC_CACHE[key] = {
            cid.strip(): (float(ra), float(dec))
            for cid, (ra, dec) in zip(atlas_data['COADD_ID'], atlas_data['CRVAL'])
        }
    return _ATLAS_RADEC_CACHE[key]


def get_overlapping_coadds(exp_ra, exp_dec, atlas_data, margin_deg=1.7, overlap_only=True):
    with _stage('overlaps'):
        tile_coords = _atlas_tile_coords(atlas_data)
        exp_coord = SkyCoord(ra=exp_ra * u.deg, dec=exp_dec * u.deg)
        sep = tile_coords.separation(exp_coord)
        matched = atlas_data[sep <= margin_deg * u.deg]
    if overlap_only and len(matched) > 0:
        # Conservative geometric overlap test: drops tiles within the radial
        # margin that can't actually overlap this exposure, so
        # project_coadd_onto_exposure/project_unwisemask_onto_exposure don't
        # waste a resample_with_wcs call (and OverlapError) on them.
        keep = exposure_overlaps_tile(exp_ra, exp_dec,
                                      matched['CRVAL'][:, 0], matched['CRVAL'][:, 1])
        matched = matched[keep]
    return [cid.strip() for cid in matched['COADD_ID']]


@lru_cache(maxsize=12)
def _load_model_tile(coadd_id, model_dir, band_num, ra, dec):
    """
    Cache a tile's star model + canonical WCS per worker process -- every
    exposure overlapping a tile would otherwise re-read the same files
    repeatedly.

    The WCS comes from get_coadd_tile_wcs(ra, dec) -- analytically, from the
    atlas table -- not from the model file's header (see _atlas_radec_map).
    """
    model_path = os.path.join(model_dir, f"{coadd_id}.{band_num}.mod.fits")
    if not os.path.exists(model_path):
        return None

    # ext=1 is full model (stars + sky); ext=2 is sky
    with _stage('model_read'):
        ext_model = fitsio.read(model_path, ext=1)
        ext_sky = fitsio.read(model_path, ext=2)
    stars_nanomag = ext_model.astype(np.float64) - ext_sky.astype(np.float64)
    return stars_nanomag, get_coadd_tile_wcs(ra, dec)


def project_coadd_onto_exposure(coadd_id, model_dir, wcs_exp, shape_out, band_num, atlas_data):
    ra, dec = _atlas_radec_map(atlas_data)[coadd_id]
    cached = _load_model_tile(coadd_id, model_dir, band_num, ra, dec)
    if cached is None:
        print(f"  [destripe] model not found: {os.path.join(model_dir, coadd_id + '.' + str(band_num) + '.mod.fits')}")
        return None
    stars_nanomag, wcs_mod = cached

    try:
        with _stage('model_resample'):
            Yo, Xo, Yi, Xi, rims = resample_with_wcs(wcs_exp, wcs_mod, [stars_nanomag], 3)
    except OverlapError:
        return None

    stars_proj = np.full(shape_out, np.nan, dtype=np.float64)
    stars_proj[Yo, Xo] = np.asarray(rims[0], dtype=np.float64)
    return stars_proj


@lru_cache(maxsize=12)
def _load_unwise_mask_tile(coadd_id, release, ra, dec):
    """Cache a tile's artifact mask groups + canonical WCS per worker process (see _load_model_tile)."""
    msk_path = (
        f"/global/cfs/cdirs/cosmo/work/wise/outputs/merge/{release}/fulldepth/"
        f"{coadd_id[:3]}/{coadd_id}/unwise-{coadd_id}-msk.fits.gz"
    )
    if not os.path.exists(msk_path):
        return None

    with _stage('mask_read'):
        msk_data = fitsio.read(msk_path)
    msk_data = msk_data & ~np.int32(1 << 6)   # drop tile-boundary bit

    # Build the three group masks as clean booleans BEFORE reprojection,
    # so the float32 cast below carries only exact 0.0/1.0 values
    psf_bool = ((msk_data & _PSF_BITS) != 0).astype(np.float32)
    galaxy_bool = ((msk_data & _GALAXY_BIT) != 0).astype(np.float32)
    bigobj_bool = ((msk_data & _BIGOBJ_BIT) != 0).astype(np.float32)
    return psf_bool, galaxy_bool, bigobj_bool, get_coadd_tile_wcs(ra, dec)


def project_unwisemask_onto_exposure(coadd_id, release, wcs_exp, shape_out, atlas_data):
    ra, dec = _atlas_radec_map(atlas_data)[coadd_id]
    cached = _load_unwise_mask_tile(coadd_id, release, ra, dec)
    if cached is None:
        return None
    psf_bool, galaxy_bool, bigobj_bool, wcs_mod = cached

    try:
        with _stage('mask_resample'):
            Yo, Xo, Yi, Xi, rims = resample_with_wcs(
                wcs_exp, wcs_mod, [psf_bool, galaxy_bool, bigobj_bool], 0
            )
    except OverlapError:
        return None

    psf_proj = np.zeros(shape_out, dtype=bool)
    galaxy_proj = np.zeros(shape_out, dtype=bool)
    bigobj_proj = np.zeros(shape_out, dtype=bool)
    psf_proj[Yo, Xo] = np.asarray(rims[0], dtype=np.float32) > 0.5
    galaxy_proj[Yo, Xo] = np.asarray(rims[1], dtype=np.float32) > 0.5
    bigobj_proj[Yo, Xo] = np.asarray(rims[2], dtype=np.float32) > 0.5

    return psf_proj, galaxy_proj, bigobj_proj


def get_unwisemask_for_exposure(coadd_ids, release, wcs_exp, shape_out, atlas_data):
    psf_combined = np.zeros(shape_out, dtype=bool)
    galaxy_combined = np.zeros(shape_out, dtype=bool)
    bigobj_combined = np.zeros(shape_out, dtype=bool)

    for coadd_id in coadd_ids:
        result = project_unwisemask_onto_exposure(coadd_id, release, wcs_exp, shape_out, atlas_data)
        if result is None:
            continue
        psf_proj, galaxy_proj, bigobj_proj = result
        psf_combined |= psf_proj
        galaxy_combined |= galaxy_proj
        bigobj_combined |= bigobj_proj

    combined = (
        binary_dilation(psf_combined, iterations=_PSF_DILATION) |
        binary_dilation(galaxy_combined, iterations=_GALAXY_DILATION) |
        binary_dilation(bigobj_combined, iterations=_BIGOBJ_DILATION)
    )
    return combined


def estimate_sky(residual, goodmask, star_mask, unwise_mask, npix=100):
    """
    Estimate 2D spatially varying sky from residual image.
    Uses full mask (goodmask + star_mask + unwise_mask) to exclude
    bright stars, galaxy wings, and artifact regions before fitting.
    """
    sky_im = get_sky_im()
    sky_weight = np.ones(residual.shape, dtype='f4')
    if goodmask is not None:
        sky_weight[~goodmask] = 0.

    sky_weight[star_mask] = 0.
    sky_weight[unwise_mask] = 0.
    sky_weight[~np.isfinite(residual)] = 0.

    sky_map = sky_im(residual.astype('f4'), weight=sky_weight, npix=npix, order=1)
    return sky_map


def _smooth(a, sigma, truncate=4.0, min_fft_sigma=8.0):
    """Gaussian smoothing, equivalent to scipy's gaussian_filter.

    mask_stars smooths with sigma = 100/2.355 and again with four times that.
    At sigma=170 on a 1016-pixel frame scipy builds a 1361-tap kernel, wider
    than the image, and evaluates it directly at every pixel; those two calls
    alone were two thirds of the whole destripe runtime.

    Convolution is far cheaper in the Fourier domain at this kernel size. Using
    scipy's own sampled-and-normalised kernel and symmetric padding (numpy's
    'symmetric' is scipy's 'reflect') reproduces gaussian_filter to ~2e-15,
    i.e. float round-off, so the 5-sigma mask thresholds downstream cannot
    flip. Small sigma falls back to the direct filter, where FFT overhead would
    not pay for itself.

    The input must be finite: a single NaN would contaminate the entire output
    via the transform, so non-finite input takes the direct path instead.
    """
    if sigma < min_fft_sigma or not np.all(np.isfinite(a)):
        return gaussian_filter(a, sigma=sigma, truncate=truncate)

    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kern = np.exp(-0.5 / (sigma * sigma) * x * x)
    kern /= kern.sum()

    out = a.astype(np.float64, copy=True)
    for axis in (0, 1):
        out = np.moveaxis(out, axis, -1)
        n = out.shape[-1]
        padded = np.pad(out, [(0, 0)] * (out.ndim - 1) + [(radius, radius)],
                        mode='symmetric')
        length = next_fast_len(padded.shape[-1] + kern.size - 1)
        spec = rfft(padded, length, axis=-1) * rfft(kern, length)
        out = np.moveaxis(irfft(spec, length, axis=-1)[..., 2 * radius:2 * radius + n],
                          -1, axis)
    return out


def mask_stars(flat_residual, star_mask, unwise_mask, goodmask=None, k_sigma=5.0, rms_fwhm=100, rms_threshold=3):

    work = flat_residual.copy()

    if goodmask is not None:
        work[~goodmask] = np.nan

    work[star_mask] = np.nan

    if unwise_mask is not None:
        work[unwise_mask] = np.nan

    # Compact outliers: > k robust sigma from this exposure's own median
    good = work[np.isfinite(work)]
    center = np.median(good)
    sigma = mad_std(good)
    bright_locs = np.abs(work - center) > k_sigma * sigma
    bright_mask = binary_dilation(bright_locs, iterations=2)
    work[bright_mask] = np.nan

    # # Gaussian filter on work
    work_rms = convolve(work, Box2DKernel(4), nan_treatment='interpolate', boundary='fill', fill_value=np.nan, preserve_nan=False)

    # --- Local statistics setup ---
    valid = np.isfinite(work_rms).astype(float)
    work_filled = np.where(np.isfinite(work_rms), work_rms, 0.0)
    sigma_rms = rms_fwhm / 2.355

    sm_mask = _smooth(valid, sigma_rms)
    sm_mean = _smooth(work_filled, sigma_rms)
    sm_mean2 = _smooth(work_filled**2, sigma_rms)

    # np.where evaluates both branches, so divide only where the denominator
    # is usable rather than computing 0/0 and discarding it.
    ok = sm_mask > 0.1
    mean_map = np.zeros_like(sm_mean)
    np.divide(sm_mean, sm_mask, out=mean_map, where=ok)
    mean2_map = np.zeros_like(sm_mean2)
    np.divide(sm_mean2, sm_mask, out=mean2_map, where=ok)
    rms_map = np.full_like(sm_mean, np.nan)
    np.sqrt(np.maximum(mean2_map - mean_map**2, 0.0), out=rms_map, where=ok)

    # --- Cut 1: high local scatter ---
    rms_flag = np.isfinite(rms_map) & (rms_map > rms_threshold)
    work[rms_flag] = np.nan

    # --- Cut 2: smooth coherent excess, hysteresis ---
    broad_sigma = 4.0 * sigma_rms
    sm_mask_b = _smooth(valid, broad_sigma)
    sm_mean_b = _smooth(work_filled, broad_sigma)
    mean_map_ambient = np.zeros_like(sm_mean_b)
    np.divide(sm_mean_b, sm_mask_b, out=mean_map_ambient, where=sm_mask_b > 0.1)
    mean_map_hp = mean_map - mean_map_ambient

    _, hp_center, hp_noise = sigma_clipped_stats(
        mean_map_hp[sm_mask > 0.5], sigma=3.0, maxiters=5
    )
    n_sigma_mean = 5.0
    n_sigma_grow = 2.5

    sig_map = np.zeros_like(mean_map_hp)
    valid_hp = np.isfinite(mean_map_hp)
    sig_map[valid_hp] = (mean_map_hp[valid_hp] - hp_center) / hp_noise

    seed_mask = (sm_mask > 0.3) & (np.abs(sig_map) > n_sigma_mean)
    grow_mask = (sm_mask > 0.3) & (np.abs(sig_map) > n_sigma_grow)

    labeled, n_labels = label(grow_mask)
    seed_labels = np.unique(labeled[seed_mask])
    seed_labels = seed_labels[seed_labels != 0]
    coherent_excess = np.isin(labeled, seed_labels)
    work[coherent_excess] = np.nan
    return work, star_mask | unwise_mask, bright_locs, rms_flag, coherent_excess


def apply_lane_correction(work_res, flat_residual, band, unwise_mask=None, hp_sigma=20):
    """
    Lane correction only: removes stripes pattern per 64-pixel channel.
    W1: collapses across 64 columns per channel -> corrects along rows
    W2: collapses across 64 rows per channel -> corrects along columns

    work_res  -- star masked residual, used for ESTIMATING the correction
    flat_residual -- unmasked residual, correction is APPLIED here
    Followed by a high pass filter to remove residual large-scale structure.
    """
    output = flat_residual.copy()   # apply correction to unmasked
    lane_width = 64
    n_lanes = 16
    smooth_sigma = 3.0
    short_axis = 1 if band == 1 else 0

    for i in range(n_lanes):
        s = i * lane_width
        e = min((i + 1) * lane_width, 1016)
        lane_width = e - s  # last lane is < 64
        slc = (slice(None), slice(s, e)) if band == 1 else (slice(s, e), slice(None))

        # Zero out positions with too few valid pixels -- sparse rows/cols
        # occur when a large galaxy dominates the lane. The median of <10
        # pixels is unreliable and can bias the correction.
        n_valid = np.sum(np.isfinite(work_res[slc]), axis=short_axis)
        work_slice = work_res[slc].copy()
        if band == 1:
            work_slice[n_valid < 0.4 * lane_width, :] = np.nan   # mask sparse rows
        else:
            work_slice[:, n_valid < 0.4 * lane_width] = np.nan   # mask sparse columns

        # Estimate from star-masked residual
        pattern = np.nanmedian(work_slice, axis=short_axis)
        sparse_mask = ~np.isfinite(pattern)

        if sparse_mask.any() and (~sparse_mask).sum() > 2:
            x = np.arange(len(pattern))
            pattern = np.interp(x, x[~sparse_mask], pattern[~sparse_mask])
        else:
            pattern = np.nan_to_num(pattern, nan=0.0)
        cc = gaussian_filter1d(pattern, sigma=smooth_sigma)

        cc_highpass = cc - gaussian_filter1d(cc, sigma=hp_sigma)
        cc_highpass[sparse_mask] = 0.0  # no correction at sparse position

        # Apply to unmasked flat residual
        if band == 1:
            output[slc] -= cc_highpass[:, None]
        else:
            output[slc] -= cc_highpass[None, :]

    return output


def apply_channel_offset_correction(work_res, flat_residual, band,
                                    min_valid_frac=0.2, max_offset=None):
    """
    Remove a constant offset per 64-pixel readout channel.

    apply_lane_correction high-passes along each lane, which also discards
    that lane's mean, and estimate_sky fits a smooth surface that cannot
    follow sharp channel edges. The per-channel DC level therefore survives
    both stages and shows up as 64-pixel banding in the residual.

    work_res      -- star-masked residual (post lane correction), for ESTIMATING
    flat_residual -- unmasked residual (post lane correction), correction APPLIED here

    Offsets are forced to sum to zero, so the frame's overall level -- which
    the coadd's sky subtraction and background matching depend on -- is left
    alone and only the relative channel-to-channel pattern is removed.

    Returns (corrected, offsets).
    """
    output = flat_residual.copy()
    lane_width = 64
    n_lanes = 16
    offsets = np.zeros(n_lanes)
    npix = np.zeros(n_lanes)
    valid = np.zeros(n_lanes, dtype=bool)
    long_axis_len = work_res.shape[0 if band == 1 else 1]

    for i in range(n_lanes):
        s = i * lane_width
        e = min((i + 1) * lane_width, 1016)
        slc = (slice(None), slice(s, e)) if band == 1 else (slice(s, e), slice(None))
        npix[i] = (e - s) * long_axis_len
        vals = work_res[slc]
        vals = vals[np.isfinite(vals)]
        # A channel dominated by a large extended source is exactly the case
        # the median cannot survive, so require most of it to be usable.
        if vals.size < min_valid_frac * npix[i]:
            continue
        offsets[i] = np.median(vals)
        valid[i] = True

    if valid.sum() < 2:
        return output, np.zeros(n_lanes)

    offsets[~valid] = 0.0
    # Weight by channel area, not channel count: 1016 = 15*64 + 56, so the
    # last channel is narrower and an unweighted mean would still shift the
    # frame's overall level.
    offsets[valid] -= (np.sum(offsets[valid] * npix[valid]) / np.sum(npix[valid]))
    if max_offset is not None:
        offsets = np.clip(offsets, -max_offset, max_offset)

    for i in range(n_lanes):
        if not valid[i]:
            continue
        s = i * lane_width
        e = min((i + 1) * lane_width, 1016)
        slc = (slice(None), slice(s, e)) if band == 1 else (slice(s, e), slice(None))
        output[slc] -= offsets[i]
    return output, offsets


def _fin(x):
    x = float(x)
    return x if np.isfinite(x) else _SENTINEL


def _collect_stats(status, weight, work_res=None, sky_map=None, lane_corr=None, work_range=np.nan,
                   ch_offsets=None):
    """Per-exposure summary numbers, header-safe."""
    return {
        "BGSTAT": status,                                              # OK / NOPROJ / BIGSKY
        "BGWRANGE": _fin(work_range),                                  # p95 - p5 of work_res [DN]
        "BGMASKF": _fin(np.isnan(work_res).mean()) if work_res is not None else _SENTINEL,
        "BGSKYMED": _fin(np.nanmedian(sky_map)) if sky_map is not None else _SENTINEL,
        "BGSKYRNG": _fin(np.nanpercentile(sky_map, 95) - np.nanpercentile(sky_map, 5))
                    if sky_map is not None else _SENTINEL,             # sky gradient [DN]
        "BGLANEAM": _fin(np.nanstd(lane_corr)) if lane_corr is not None else _SENTINEL,  # applied stripe amplitude [DN]
        "BGCOVF": _fin((weight > 0).mean()) if weight is not None else _SENTINEL,        # coadd model coverage
        "BGCHOFF": _fin(np.std(ch_offsets)) if ch_offsets is not None else _SENTINEL,    # channel offset spread [DN]
    }


def failed_stats():
    """Stats for a frame where the correction raised an exception."""
    return _collect_stats("FAILED", None)


def destripe_frame(fullimg, fullgood, wcs_exp, zp, ra_exp, dec_exp, atlas_data,
                    model_dir, band_num, release,
                    star_dn_threshold=30.0, dilation_iters=4, unwise_dilation_iters=1,
                    sky_gradient_limit=300.0, correct_channels=False):
    """
    Correct one full L1b frame already loaded in memory (no file I/O for the
    frame itself -- the caller, unwise_coadd.py's round 1, has already read
    int/unc/msk). `fullgood` is the caller's own hardware-mask-derived
    good-pixel map (same maskbits used for round 1's own goodmask), so the
    correction's pixel rejection is identical to round 1's.

    Returns (corrected_image, stats). `corrected_image` is `fullimg`
    unmodified (same array) for the NOPROJ case (no overlapping model found
    -- e.g. no coadd tile has been built yet for this sky position) and for
    the BIGSKY case (residual sky gradient too large to trust the
    correction -- likely a bright extended source or moonlight); only the
    OK case returns an actually-corrected image.
    """
    shape_out = fullimg.shape
    coadd_ids = get_overlapping_coadds(ra_exp, dec_exp, atlas_data)

    sum_stars = np.zeros(shape_out, dtype=np.float64)
    weight = np.zeros(shape_out, dtype=np.float64)
    for coadd_id in coadd_ids:
        stars_proj = project_coadd_onto_exposure(coadd_id, model_dir, wcs_exp, shape_out, band_num, atlas_data)
        if stars_proj is None:
            continue
        valid = np.isfinite(stars_proj)
        sum_stars[valid] += stars_proj[valid]
        weight[valid] += 1.0

    if weight.max() == 0:
        return fullimg, _collect_stats("NOPROJ", weight)

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_stars_nm = np.where(weight > 0, sum_stars / weight, np.nan)

    # Convert the model (nanomaggies, zeropoint 22.5) to this frame's DN,
    # using the exact same zp (and formula) unwise_coadd.py's round 1 used
    # to compute `zp` in the first place -- see unwise_utils.zeropointToScale.
    stars_dn = avg_stars_nm * zeropointToScale(zp)
    residual = fullimg - stars_dn

    unwise_flagged = get_unwisemask_for_exposure(coadd_ids, release, wcs_exp, shape_out, atlas_data)

    with _stage('dilate'):
        star_locs = (stars_dn > star_dn_threshold)
        star_mask = binary_dilation(star_locs, iterations=dilation_iters)
        unwise_mask = binary_dilation(unwise_flagged, iterations=unwise_dilation_iters)

    with _stage('sky'):
        sky_map = estimate_sky(residual, fullgood, star_mask, unwise_mask, npix=100)
    flat_residual = residual - sky_map

    with _stage('mask_stars'):
        work_res, combined_mask, _, _, _ = mask_stars(
            flat_residual, star_mask=star_mask, unwise_mask=unwise_mask, goodmask=fullgood,
        )

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        work_range = np.nanpercentile(work_res, 95) - np.nanpercentile(work_res, 5)

    if work_range > sky_gradient_limit:
        return fullimg, _collect_stats("BIGSKY", weight, work_res, sky_map, None, work_range)

    with _stage('lane'):
        corr_work_res = apply_lane_correction(work_res, flat_residual, band_num, unwise_mask)

    ch_offsets = None
    if correct_channels:
        # Estimate from the lane-corrected residual under the same mask that
        # work_res used, and apply to the unmasked lane-corrected residual.
        with _stage('chan'):
            masked_corr = np.where(np.isfinite(work_res), corr_work_res, np.nan)
            corr_work_res, ch_offsets = apply_channel_offset_correction(
                masked_corr, corr_work_res, band_num)

    corrected = corr_work_res + stars_dn + sky_map

    nan_mask = ~np.isfinite(corrected)
    corrected[nan_mask] = fullimg[nan_mask]

    stats = _collect_stats("OK", weight, work_res, sky_map, flat_residual - corr_work_res, work_range,
                           ch_offsets=ch_offsets)
    _maybe_report_timing()
    return corrected.astype(fullimg.dtype), stats


def corrected_frame_path(savedir, intfn):
    """Mirror unwise_utils.get_l1b_file's directory layout under `savedir`,
    from an existing L1b path: <savedir>/<scan_id[-2:]>/<scan_id>/<frame_num>/<basename>."""
    base = os.path.basename(intfn).replace('.gz', '')
    scan_id = base[:6]
    frame_dir = base[6:9]
    return os.path.join(savedir, scan_id[-2:], scan_id, frame_dir, base)


def save_corrected_exposure(corrected_exposure, hdr_exp, output_path, stats=None,
                            correct_channels=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if stats is not None:
        for k, v in stats.items():
            hdr_exp.add_record({"name": k, "value": v})
    hdr_exp.add_record({"name": "UNW_DSTV", "value": DESTRIPE_VERSION,
                        "comment": "unWISE stripe-correction version"})
    hdr_exp.add_record({"name": "UNW_DSTC", "value": bool(correct_channels),
                        "comment": "per-channel offset correction applied?"})
    tmp = f"{output_path}.{os.uname().nodename}.{os.getpid()}.tmp"
    fitsio.write(tmp, corrected_exposure.astype(np.float32), header=hdr_exp, clobber=True)
    os.replace(tmp, output_path)


def load_corrected_exposure(path, shape=None, correct_channels=False):
    """
    Read back a corrected exposure written by save_corrected_exposure, for use
    as a cache entry. Returns (image, stats), or None if the file is absent,
    unreadable, the wrong shape, was not written by DESTRIPE_VERSION, or was
    corrected with a different --destripe-channels setting.

    The correction depends only on per-frame inputs, so a cached frame is valid
    for any tile. Returning None simply means "recompute", so every failure
    mode here is non-fatal by construction.
    """
    try:
        img, hdr = fitsio.read(path, header=True)
    except Exception:
        return None
    try:
        if int(hdr["UNW_DSTV"]) != DESTRIPE_VERSION:
            return None
        if bool(hdr["UNW_DSTC"]) != bool(correct_channels):
            return None
        if shape is not None and img.shape != shape:
            return None
        stats = {}
        for k in STATS_KEYS:
            v = hdr[k]
            stats[k] = v.strip() if isinstance(v, str) else float(v)
        # only fully corrected frames are saved, so anything else is suspect
        if stats["BGSTAT"] != "OK":
            return None
    except Exception:
        return None
    return img.astype(np.float32), stats
