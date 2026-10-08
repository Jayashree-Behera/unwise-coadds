import os
import sys
import numpy as np
import fitsio
import matplotlib
matplotlib.use('Agg')  # must be set before importing pyplot; this pipeline runs headless
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore") #category=RuntimeWarning)

import yaml
import tempfile
import shutil
import gzip
import logging

sys.path.append("/pscratch/sd/s/shreeb/shreeb/")
sys.path.append("/pscratch/sd/s/shreeb/shreeb/unwise-coadds")
sys.path.append("/pscratch/sd/s/shreeb/shreeb/crowdsource/crowdsource")
from crowdsource import wise_proc, crowdsource_base
from crowdsource_base import sky_im

import unwise_coadd
from unwise_utils import phase_from_scanid, good_scan_mask, get_coadd_tile_wcs

from astropy.io import fits
from astropy.coordinates import SkyCoord 
import astropy.units as u
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.convolution import convolve, Box2DKernel


from astrometry.util.util import Sip
from astrometry.util.resample import resample_with_wcs, OverlapError

from scipy.ndimage import binary_dilation, gaussian_filter1d, gaussian_filter, label

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from functools import lru_cache
import argparse


def existing_path(path):
    """Return `path` or `path + '.gz'`, whichever exists on disk, else None."""
    if os.path.exists(path):
        return path
    gz = path + '.gz'
    return gz if os.path.exists(gz) else None


def _resolve_medfilt(medfilt, band_num):
    """unwise_coadd.py's own resolution of --medfilt: None -> 50 for W3,W4 else 0."""
    return medfilt if medfilt is not None else (50 if band_num in (3, 4) else 0)



 
 
_TILE_W      = 2048
_TILE_PS     = 2.75 / 3600.0          # unWISE coadd pixel scale [deg/px]
_EXP_HALFDIAG = 1016 * 2.75 / 3600.0 * np.sqrt(2) / 2   # ≈ 0.549 deg
_OVERLAP_PAD  = _EXP_HALFDIAG / _TILE_PS + 20            # ≈ 738 px, +20 px safety for SIP/rounding

def exposure_overlaps_tile(exp_ra, exp_dec, tile_ra, tile_dec):
    """
    Vectorized, conservative test: True if an exposure centred at (exp_ra, exp_dec)
    can overlap the 2048x2048 unWISE tile centred at (tile_ra, tile_dec), for any
    exposure rotation. Gnomonic projection onto the tile's tangent plane, then
    check against the tile box expanded by the exposure half-diagonal.
    Never drops an exposure that touches the tile; keeps a few that don't.
    """
    a  = np.radians(np.asarray(exp_ra, dtype=float))
    d  = np.radians(np.asarray(exp_dec, dtype=float))
    a0 = np.radians(tile_ra)
    d0 = np.radians(tile_dec)

    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi  = np.cos(d) * np.sin(a - a0) / cosc
        eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc

    # unWISE tile WCS: CRPIX = (W+1)/2, CD1_1 = -ps, CD2_2 = +ps
    crpix = (_TILE_W + 1) / 2.0
    x = crpix - np.degrees(xi)  / _TILE_PS
    y = crpix + np.degrees(eta) / _TILE_PS

    lo, hi = 0.5 - _OVERLAP_PAD, _TILE_W + 0.5 + _OVERLAP_PAD
    return (cosc > 0) & (x > lo) & (x < hi) & (y > lo) & (y < hi)
    

def get_exposures_for_tile(tile_ra, tile_dec, band_str, meta_dir, radius=1.7, overlap_only=True):
    index_path = os.path.join(meta_dir, f'WISE-index-L1b_{band_str}.fits')
    cat = fitsio.read(index_path)

    with open(os.path.join(meta_dir, 'l1b_dirs.yml'), 'r') as f:
        l1b_dirs = yaml.safe_load(f)

    exp_coords = SkyCoord(ra=cat['RA'] * u.deg, dec=cat['DEC'] * u.deg)
    tile_coord = SkyCoord(ra=tile_ra * u.deg,   dec=tile_dec * u.deg)
    keep = exp_coords.separation(tile_coord) <= radius * u.deg

    if overlap_only:
        n_circle = keep.sum()
        keep &= exposure_overlaps_tile(cat['RA'], cat['DEC'], tile_ra, tile_dec)
        print(f"    overlap filter: {n_circle} -> {keep.sum()} exposures")

    matches = cat[keep]
    paths = []
    for row in matches:
        phase = phase_from_scanid(row['SCAN_ID'])
        wdir  = l1b_dirs[phase]
        paths.append(os.path.join(
            wdir,
            f"{row['SCANGRP']}/{row['SCAN_ID']}/{row['FRAME_NUM']:03d}/"
            f"{row['SCAN_ID']}{row['FRAME_NUM']:03d}-{band_str}-int-1b.fits"
        ))
    return paths


def tile_center(cid):
    """(ra, dec) in deg from a coadd_id, e.g. 2709p666 -> (270.9, 66.6)."""
    ra = int(cid[:4]) / 10.0
    dec = int(cid[-3:]) / 10.0 * (1 if cid[4] == "p" else -1)
    return ra, dec
 
 
def find_valid_paths(tiles, band_str, meta_dir, radius=1.7, overlap_only=True, nthreads=32):
    """
    Union of exposures over all tiles (overlapping exposures processed once),
    then keep those present on disk (plain or .gz).

    Returns
    -------
    valid     : list of existing exposure paths (may end in .gz)
    tile_sets : {coadd_id: set of index paths}, many-to-many membership,
                used to group results per tile for plotting
    """
    all_paths, seen, tile_sets = [], set(), {}
    for cid in tiles:
        ra, dec = tile_center(cid)
        paths = get_exposures_for_tile(ra, dec, band_str, meta_dir, radius=radius,
                                       overlap_only=overlap_only)
        tile_sets[cid] = set(paths)
        new = [q for q in paths if q not in seen]
        seen.update(new)
        all_paths.extend(new)
        print(f"  {cid}: {len(paths)} exposures ({len(new)} new)")

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        valid = [r for r in ex.map(existing_path, all_paths) if r is not None]
    print(f"  Unique: {len(all_paths)}, on disk: {len(valid)}")
    return valid, tile_sets


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
    tile_coords = _atlas_tile_coords(atlas_data)
    exp_coord = SkyCoord(ra=exp_ra * u.deg, dec=exp_dec * u.deg)
    sep  = tile_coords.separation(exp_coord)
    matched = atlas_data[sep <= margin_deg * u.deg]
    if overlap_only and len(matched) > 0:
        # Conservative geometric overlap test (same one used in
        # find_valid_paths/get_exposures_for_tile): drops tiles within the
        # radial margin that can't actually overlap this exposure, so
        # project_coadd_onto_exposure/project_unwisemask_onto_exposure don't
        # waste a resample_with_wcs call (and OverlapError) on them.
        keep = exposure_overlaps_tile(exp_ra, exp_dec,
                                      matched['CRVAL'][:, 0], matched['CRVAL'][:, 1])
        matched = matched[keep]
    return [cid.strip() for cid in matched['COADD_ID']]


def load_exposure(exposure_path):
    path = exposure_path if os.path.exists(exposure_path) else exposure_path + '.gz'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Exposure not found: {exposure_path}")

    if path.endswith('.gz'):
        with tempfile.NamedTemporaryFile(suffix='.fits') as tmp:
            with gzip.open(path, 'rb') as f_in:
                shutil.copyfileobj(f_in, tmp)
            tmp.flush()
            wcs = Sip(tmp.name)
            data, hdr = fitsio.read(tmp.name, header=True)
    else:
        wcs = Sip(path)
        data, hdr = fitsio.read(path, header=True)

    wcs.set_crval(np.array([hdr['CRVAL1'], hdr['CRVAL2']]))
    return wcs, data.astype(np.float64), hdr



@lru_cache(maxsize=12)
def _load_model_tile(coadd_id, model_dir, band_num, ra, dec):
    """
    Cache a tile's star model + canonical WCS per worker process -- every
    exposure overlapping a tile would otherwise re-read (and for masks,
    re-gunzip) the same handful of files repeatedly.

    The WCS comes from get_coadd_tile_wcs(ra, dec) -- analytically, from the
    atlas table -- not from the model file's header (see _atlas_radec_map).
    """
    model_path = os.path.join(model_dir, f"{coadd_id}.{band_num}.mod.fits")
    if not os.path.exists(model_path):
        return None

    # ext=1 is full model (stars + sky);  ext=2 is sky
    ext_model = fitsio.read(model_path, ext=1)
    ext_sky = fitsio.read(model_path, ext=2)
    stars_nanomag = ext_model.astype(np.float64) - ext_sky.astype(np.float64)
    return stars_nanomag, get_coadd_tile_wcs(ra, dec)


def project_coadd_onto_exposure(coadd_id, model_dir, wcs_exp, shape_out, band_num, atlas_data):
    ra, dec = _atlas_radec_map(atlas_data)[coadd_id]
    cached = _load_model_tile(coadd_id, model_dir, band_num, ra, dec)
    if cached is None:
        print(f"  [project] model not found: {os.path.join(model_dir, coadd_id + '.' + str(band_num) + '.mod.fits')}")
        return None
    stars_nanomag, wcs_mod = cached

    # Single resample_with_wcs call projects the model (fast)
    try:
        Yo, Xo, Yi, Xi, rims = resample_with_wcs(wcs_exp, wcs_mod, [stars_nanomag], 3)
    except OverlapError:
        return None

    stars_proj = np.full(shape_out, np.nan, dtype=np.float64)
    stars_proj[Yo, Xo] = np.asarray(rims[0], dtype=np.float64)
    return stars_proj



def average_coadds_for_exposure(exposure_path, atlas_data, model_dir, band_num):
    wcs_exp, data_exp, hdr_exp = load_exposure(exposure_path)
    shape_out = data_exp.shape

    coadd_ids = get_overlapping_coadds(float(hdr_exp['CRVAL1']), float(hdr_exp['CRVAL2']), atlas_data)
 
    sum_stars = np.zeros(shape_out, dtype=np.float64)
    # sum_sky = np.zeros(shape_out, dtype=np.float64)
    weight = np.zeros(shape_out, dtype=np.float64)
 
    for coadd_id in coadd_ids:
        stars_proj = project_coadd_onto_exposure(
            coadd_id, model_dir, wcs_exp, shape_out, band_num, atlas_data
        )
        if stars_proj is None:
            continue
        valid = np.isfinite(stars_proj) # & np.isfinite(sky_proj)
        sum_stars[valid] += stars_proj[valid]
        # sum_sky  [valid] += sky_proj  [valid]
        weight[valid] += 1.0

    if weight.max() == 0:
        return None, data_exp, hdr_exp, weight, coadd_ids, wcs_exp
 
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_stars_nanomag = np.where(weight > 0, sum_stars / weight, np.nan)
        # avg_sky_nanomag   = np.where(weight > 0, sum_sky   / weight, np.nan)
 
    return avg_stars_nanomag, data_exp, hdr_exp, weight, coadd_ids, wcs_exp #avg_sky_nanomag, 


def construct_residual_exposure(exposure_path, atlas_data, model_dir, band_num, zp_lookup=None):
    
    avg_stars_nm, data_exp, hdr_exp, weight, coadd_ids, wcs_exp = average_coadds_for_exposure(
        exposure_path, atlas_data, model_dir, band_num
    )

    if avg_stars_nm is None:
        return data_exp, None, None, hdr_exp, weight, coadd_ids, wcs_exp

    # Zeropoint: prefer zp_lookup (a zp_lookup.ZPLookUp, matching how
    # unwise_coadd.py itself derives zp during round 1 -- see its
    # use_zp_meta / ZPLookUp.get_zp logic) over the raw MAGZP header card,
    # so the stripe correction's photometric scaling matches the coadd's.
    if zp_lookup is not None:
        zp = zp_lookup.get_zp(hdr_exp['MJD_OBS'])
    else:
        zp = float(hdr_exp['MAGZP'])
    zpscale  = 10.0 ** ((22.5 - zp) / 2.5)
    stars_dn = avg_stars_nm / zpscale
    residual = data_exp - stars_dn
 
    return data_exp, stars_dn, residual, hdr_exp, weight, coadd_ids, wcs_exp



# Unwise flag bit group
_PSF_BITS = sum(1<<b for b in [0,1,2,3,4,5,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
_GALAXY_BIT  = (1 << 9)
_BIGOBJ_BIT  = (1 << 10)

_PSF_DILATION = 1
_GALAXY_DILATION = 6
_BIGOBJ_DILATION = 6


@lru_cache(maxsize=12)
def _load_unwise_mask_tile(coadd_id, release, ra, dec):
    """Cache a tile's artifact mask groups + canonical WCS per worker process (see _load_model_tile)."""
    msk_path = (
        f"/global/cfs/cdirs/cosmo/work/wise/outputs/merge/{release}/fulldepth/"
        f"{coadd_id[:3]}/{coadd_id}/unwise-{coadd_id}-msk.fits.gz"
    )
    if not os.path.exists(msk_path):
        return None

    msk_data = fitsio.read(msk_path)
    msk_data = msk_data & ~np.int32(1 << 6)   # drop tile-boundary bit

    # Build the three group masks as clean booleans BEFORE reprojection,
    # so the float32 cast below carries only exact 0.0/1.0 values
    psf_bool = ((msk_data & _PSF_BITS)   != 0).astype(np.float32)
    galaxy_bool = ((msk_data & _GALAXY_BIT) != 0).astype(np.float32)
    bigobj_bool = ((msk_data & _BIGOBJ_BIT) != 0).astype(np.float32)
    return psf_bool, galaxy_bool, bigobj_bool, get_coadd_tile_wcs(ra, dec)


def project_unwisemask_onto_exposure(coadd_id, release, wcs_exp, shape_out, atlas_data):
    ra, dec = _atlas_radec_map(atlas_data)[coadd_id]
    cached = _load_unwise_mask_tile(coadd_id, release, ra, dec)
    if cached is None:
        return None
    psf_bool, galaxy_bool, bigobj_bool, wcs_mod = cached

    # Resample all three group masks in a single call -- all exactly 0/1
    try:
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
        binary_dilation(psf_combined, iterations=_PSF_DILATION)  |
        binary_dilation(galaxy_combined, iterations=_GALAXY_DILATION) |
        binary_dilation(bigobj_combined, iterations=_BIGOBJ_DILATION)
    )
    return combined


def estimate_sky(residual, goodmask, star_mask, unwise_mask, npix=100):
    """
    Estimate 2D spatially varying sky from residual image.
    Uses full mask (goodmask + star_mask + unwise_mask) to exclude
    bright stars, galaxy wings, and artifact regions before fitting.
    
    Parameters
    ----------
    residual    : data - stars_dn, before sky subtraction
    goodmask    : True where pixels are good (hardware mask)
    star_mask   : True where stars/galaxies are masked
    unwise_mask : True where unWISE artifact bits are set
    npix        : sky_im bin size in pixels
    
    Returns
    -------
    sky_map : 2D spatially varying sky estimate
    """
    sky_weight = np.ones(residual.shape, dtype='f4')
    if goodmask is not None:
        sky_weight[~goodmask]              = 0.
    
    sky_weight[star_mask]              = 0.
    sky_weight[unwise_mask]            = 0.
    sky_weight[~np.isfinite(residual)] = 0.

    sky_map = sky_im(residual.astype('f4'), weight=sky_weight, npix=npix, order=1)
    return sky_map


def mask_stars(flat_residual, star_mask, unwise_mask, goodmask=None, k_sigma=5.0, rms_fwhm=100, rms_threshold=3):
    
    work = flat_residual.copy()

    if goodmask is not None:
        work[~goodmask] = np.nan

    work[star_mask] = np.nan

    if unwise_mask is not None:
        work[unwise_mask] = np.nan

    # Compact outliers: > k robust sigma from this exposure's own median
    good   = work[np.isfinite(work)]
    center = np.median(good)
    sigma  = mad_std(good)
    bright_locs = np.abs(work - center) > k_sigma * sigma
    bright_mask = binary_dilation(bright_locs, iterations=2)
    work[bright_mask] = np.nan

    # Gaussian filter on work
    work_rms = convolve(work, Box2DKernel(4),nan_treatment='interpolate', boundary='fill', fill_value=np.nan, preserve_nan=False)

    # --- Local statistics setup ---
    valid       = np.isfinite(work_rms).astype(float)
    work_filled = np.where(np.isfinite(work_rms), work_rms, 0.0)
    sigma_rms   = rms_fwhm / 2.355

    sm_mask     = gaussian_filter(valid,          sigma=sigma_rms)
    sm_mean     = gaussian_filter(work_filled,    sigma=sigma_rms)
    sm_mean2    = gaussian_filter(work_filled**2, sigma=sigma_rms)

    mean_map    = np.where(sm_mask > 0.1, sm_mean / sm_mask, 0.0)
    rms_map     = np.where(
        sm_mask > 0.1,
        np.sqrt(np.maximum(sm_mean2 / sm_mask - mean_map**2, 0.0)),
        np.nan
    )

    # --- Cut 1: high local scatter ---
    rms_flag      = np.isfinite(rms_map) & (rms_map > rms_threshold)
    work[rms_flag] = np.nan

    # --- Cut 2: smooth coherent excess, hysteresis ---
    broad_sigma      = 4.0 * sigma_rms
    sm_mask_b        = gaussian_filter(valid,       sigma=broad_sigma)
    sm_mean_b        = gaussian_filter(work_filled, sigma=broad_sigma)
    mean_map_ambient = np.where(sm_mask_b > 0.1, sm_mean_b / sm_mask_b, 0.0)
    mean_map_hp      = mean_map - mean_map_ambient

    _, hp_center, hp_noise = sigma_clipped_stats(
        mean_map_hp[sm_mask > 0.5], sigma=3.0, maxiters=5
    )
    n_sigma_mean = 5.0
    n_sigma_grow = 2.5

    sig_map = np.zeros_like(mean_map_hp)
    valid_hp = np.isfinite(mean_map_hp)
    sig_map[valid_hp] = (mean_map_hp[valid_hp] - hp_center) / hp_noise   # hp_center now included

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
    W1: collapses across 64 columns per channel → corrects along rows
    W2: collapses across 64 rows per channel → corrects along columns

    work_res  — star masked residual, used for ESTIMATING the correction
    flat_residual — unmasked residual, correction is APPLIED here
    Followed by a high pass filter to remove residual large-scale structure.
    """
    output = flat_residual.copy()   # apply correction to unmasked
    lane_width  = 64
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
        sparse_mask  = ~np.isfinite(pattern)

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


def correct_single_exposure(exposure_path, atlas_data, model_dir, band_num, release, star_dn_threshold=30.0, dilation_iters=4, unwise_dilation_iters=1, zp_lookup=None):

    data_exp, stars_dn, residual, hdr_exp, weight, coadd_ids, wcs_exp = construct_residual_exposure(exposure_path, atlas_data, model_dir, band_num, zp_lookup=zp_lookup)

    if stars_dn is None:
        print(f"  SKIP (no projection): {os.path.basename(exposure_path)}")
        diagnostics = {
            'n_star_pix'        : 0,
            'work_range'        : 0.0,
            'weight'            : weight,
            'star_dn_threshold' : star_dn_threshold,
            'corrected_work_res': None,
            'skipped'           : True,
            'skip_reason'       : 'no_projection',
            'stats': _collect_stats("NOPROJ", weight),
        }
        return (data_exp, None, None, None, hdr_exp, diagnostics)

    unc_path  = existing_path(exposure_path.replace('-int-', '-unc-'))
    mask_path = existing_path(exposure_path.replace('-int-', '-msk-'))

    if unc_path is None or mask_path is None:
        raise FileNotFoundError(f"Missing -unc- or -msk- file for {exposure_path}")
        
    # Hardware masks
    unc_exp  = fitsio.read(unc_path).astype(np.float64)
    msk_data = fitsio.read(mask_path)
    badbits  = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,21,26,27,28]
    maskbits = sum(1 << b for b in badbits)
    goodmask = ((msk_data & maskbits) == 0)
    goodmask[unc_exp == 0] = False
    goodmask[~np.isfinite(data_exp)] = False
    
    # Build unWISE artifact mask in exposure frame from overlapping coadd tiles
    unwise_flagged = get_unwisemask_for_exposure(coadd_ids, release, wcs_exp, data_exp.shape, atlas_data)

    # star mask 
    star_locs   = (stars_dn > star_dn_threshold)
    star_mask   = binary_dilation(star_locs, iterations=dilation_iters)
    unwise_mask = binary_dilation(unwise_flagged, iterations=unwise_dilation_iters)

    # sky estimation with full mask
    sky_map = estimate_sky(residual, goodmask, star_mask, unwise_mask, npix=100)
    flat_residual = residual - sky_map

    # More masking
    work_res, combined_mask, _, _, _ = mask_stars(
        flat_residual,
        star_mask   = star_mask,
        unwise_mask = unwise_mask,
        goodmask    = goodmask,
    )
    n_star_pix = int(combined_mask.sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        work_range = np.nanpercentile(work_res, 95) - np.nanpercentile(work_res, 5)

    if work_range > 300.0:
        print(f"  SKIP (large sky gradient, {work_range:.1f} DN): "
              f"{exposure_path}")
        diagnostics = {
            'n_star_pix'        : n_star_pix,
            'work_range'        : work_range,
            'weight'            : weight,
            'star_dn_threshold' : star_dn_threshold,
            'corrected_work_res': work_res.copy(),
            'skipped'           : True,
            'skip_reason'       : 'large_sky_gradient',
            'stats': _collect_stats("BIGSKY", weight, work_res, sky_map, None, work_range),
        }
        return (data_exp, residual, work_res, stars_dn, hdr_exp, diagnostics)

    corr_work_res      = apply_lane_correction(work_res, flat_residual, band_num, unwise_mask)
    corrected_exposure = corr_work_res + stars_dn + sky_map

    nan_mask = ~np.isfinite(corrected_exposure)
    corrected_exposure[nan_mask] = data_exp[nan_mask]

    diagnostics = {
        'n_star_pix'        : n_star_pix,
        'work_range'        : work_range,
        'weight'            : weight,
        'star_dn_threshold' : star_dn_threshold,
        'corrected_work_res': corr_work_res.copy(),
        'skipped'           : False,
        'stats': _collect_stats("OK", weight, work_res, sky_map, flat_residual - corr_work_res, work_range),
    }

    return (corrected_exposure, residual, work_res, stars_dn, hdr_exp, diagnostics)


def plot_correction(exposure_path, corr_exp, work_res, diag, edges=False):
    """
    Two-panel diagnostic plot for one corrected L1b exposure.
    Panel 1: Raw | Corrected | Difference (lane corrections applied)
    Panel 2: Workspace column/row medians before and after lane correction
    """
    _, raw_exp, _ = load_exposure(exposure_path)
    corr_work_res = diag['corrected_work_res']
    lane_diff     = raw_exp - corr_exp
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"{os.path.basename(exposure_path)}\n"
        f"Star pix masked: {diag['n_star_pix']} ({100*diag['n_star_pix']/raw_exp.size:.1f}%)  |  "
        f"Work range: {diag['work_range']:.1f} DN  |  "
        f"Skipped: {diag['skipped']}",
        y=1.02
    )
    vlo, vhi = np.nanpercentile(raw_exp, [1, 99])
    clim      = max(0.1, np.nanpercentile(np.abs(lane_diff), 99))
    axes[0, 0].imshow(raw_exp,  vmin=vlo,   vmax=vhi,  cmap='binary',  origin='lower')
    axes[0, 0].set_title('Raw Exposure (DN)')
    axes[0, 1].imshow(corr_exp, vmin=vlo,   vmax=vhi,  cmap='binary',  origin='lower')
    axes[0, 1].set_title('Corrected Exposure (DN)')
    diff = axes[0, 2].imshow(lane_diff, vmin=-clim, vmax=clim, cmap='RdBu_r', origin='lower')
    axes[0, 2].set_title(f'Lane Correction (raw − corrected)')
    plt.colorbar(diff, ax=axes[0, 2], fraction=0.047, pad=0.02)

    if edges:
        h, w = raw_exp.shape
        lane_width = 64
        for ax in [axes[0, 0], axes[0, 1]]:
            # W1: vertical lines (column boundaries)
            for x in range(lane_width, w, lane_width):
                ax.axvline(x, color='cyan', lw=0.5, alpha=0.6, label='W1 lane' if x == lane_width else None)
            # W2: horizontal lines (row boundaries)
            for y in range(lane_width, h, lane_width):
                ax.axhline(y, color='yellow', lw=0.5, alpha=0.6, label='W2 lane' if y == lane_width else None)

    corr_work_res_masked = corr_work_res.copy()
    corr_work_res_masked[~np.isfinite(work_res)] = np.nan  # same mask as work_res
    for data, label, color in [
        (work_res,            'Before', 'C0'),
        (corr_work_res_masked,'After',  'C1'),
    ]:
        col_med = np.nanmedian(data, axis=0)
        row_med = np.nanmedian(data, axis=1)
        axes[1, 0].plot(col_med - np.nanmedian(col_med), alpha=0.7, label=label, color=color)
        axes[1, 1].plot(row_med - np.nanmedian(row_med), alpha=0.7, label=label, color=color)
    axes[1, 0].set_title('Column medians (workspace)')
    axes[1, 1].set_title('Row medians (workspace)')
    cov = axes[1, 2].imshow(diag['weight'], vmin=0, vmax=4, cmap='viridis', origin='lower')
    axes[1, 2].set_title('Coadd coverage (n tiles)')
    plt.colorbar(cov, ax=axes[1, 2], fraction=0.047, pad=0.02)
    for ax in axes[1, :2]:
        ax.axhline(0, color='k', lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for ax in axes[0].ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].set_xlabel('Column index'); axes[1, 0].set_ylabel('Median DN (recentred)')
    axes[1, 1].set_xlabel('Row index');    axes[1, 1].set_ylabel('Median DN (recentred)')
    plt.tight_layout()


def save_corrected_exposure(corrected_exposure, hdr_exp, output_path, stats=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if stats is not None:
        for k, v in stats.items():
            hdr_exp.add_record({"name": k, "value": v})
    tmp = f"{output_path}.{os.uname().nodename}.{os.getpid()}.tmp"
    fitsio.write(tmp, corrected_exposure.astype(np.float32), header=hdr_exp, clobber=True)
    os.replace(tmp, output_path)


def _atomic_copy(src, dst):
    """Copy via a unique temp file, so two jobs copying the same file can't corrupt it."""
    tmp = f"{dst}.{os.uname().nodename}.{os.getpid()}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def init_worker(atlas, model_dir, band, base_outdir, release, force=False, use_zp_meta=False):
    """
    Set up a worker process for process_one_exposure.

    use_zp_meta mirrors unwise_coadd.py's own --use_zp_meta flag: by
    default (False) zeropoints come from zp_lookup.ZPLookUp(band, poly=True)
    -- the same per-frame polynomial zeropoint unwise_coadd.py uses during
    round 1 -- rather than from the exposure's raw MAGZP header card.
    """
    global _atlas_data, _MODEL_DIR, _BAND_NUM, BASE_OUTDIR, _RELEASE, _FORCE, _ZP_LOOKUP
    _atlas_data = atlas
    _MODEL_DIR  = model_dir
    _BAND_NUM   = band
    BASE_OUTDIR = base_outdir
    _RELEASE    = release
    _FORCE      = force
    _ZP_LOOKUP  = None
    if not use_zp_meta:
        from zp_lookup import ZPLookUp
        _ZP_LOOKUP = ZPLookUp(band, poly=True)


def process_one_exposure(args):
    """
    Worker function: corrects one L1b exposure and saves output files.
    Returns (index, status_string).
    """
    idx, exposure_path = args   # back to 2-tuple, matches enumerate(valid_paths)

    path_exp  = existing_path(exposure_path)
    unc_path  = existing_path(exposure_path.replace('-int-', '-unc-')) if path_exp else None
    mask_path = existing_path(exposure_path.replace('-int-', '-msk-')) if path_exp else None

    if path_exp is None or unc_path is None or mask_path is None:
        return idx, "missing", None

    base_name    = os.path.basename(path_exp).replace('.gz', '')
    scan_id      = base_name[:6]
    final_dir    = os.path.join(BASE_OUTDIR, scan_id[-2:], scan_id, base_name[6:9])
    out_int_path = os.path.join(final_dir, base_name)
    out_unc_path = os.path.join(final_dir, base_name.replace('-int-', '-unc-') + '.gz')
    out_msk_path = os.path.join(final_dir, base_name.replace('-int-', '-msk-') + '.gz')

    if not _FORCE and all(os.path.exists(p) for p in (out_int_path, out_unc_path, out_msk_path)):
        return idx, "pre-existing", read_stats_from_header(out_int_path)

    try:
        corrected, _, _, _, hdr_exp, diag = correct_single_exposure(
            path_exp, _atlas_data, _MODEL_DIR, _BAND_NUM, _RELEASE,
            star_dn_threshold = 30.0,
            dilation_iters    = 4,
            zp_lookup         = _ZP_LOOKUP,
        )
    except Exception as e:
        return idx, f"math_failed: {e}", None

    stats = diag.get('stats')
    if stats is not None:
        stats = dict(stats, MJD=_fin(hdr_exp.get('MJD_OBS', np.nan)))

    try:
        save_corrected_exposure(corrected, hdr_exp, out_int_path,
                                stats={k: stats[k] for k in _BG_KEYS} if stats else None)
    except Exception as e:
        return idx, f"write_failed: {e}", None

    try:
        if not os.path.exists(out_unc_path): _atomic_copy(unc_path,  out_unc_path)
        if not os.path.exists(out_msk_path): _atomic_copy(mask_path, out_msk_path)
    except Exception as e:
        return idx, f"copy_failed: {e}", None

    status = "success"
    if diag['skipped']:
        reason = 'no_proj' if diag['skip_reason'] == 'no_projection' else 'large_sky'
        status = f"skipped_{reason}  range={diag['work_range']:.1f} DN"
    return idx, status, stats


_BG_KEYS = ["BGSTAT", "BGWRANGE", "BGMASKF", "BGSKYMED", "BGSKYRNG", "BGLANEAM", "BGCOVF"]
_SENTINEL = -999.0   # FITS headers cannot hold NaN


def _fin(x):
    x = float(x)
    return x if np.isfinite(x) else _SENTINEL


def _collect_stats(status, weight, work_res=None, sky_map=None, lane_corr=None, work_range=np.nan):
    """Per-exposure summary numbers, header-safe."""
    return {
        "BGSTAT"  : status,                                            # OK / NOPROJ / BIGSKY
        "BGWRANGE": _fin(work_range),                                  # p95 - p5 of work_res [DN]
        "BGMASKF" : _fin(np.isnan(work_res).mean()) if work_res is not None else _SENTINEL,
        "BGSKYMED": _fin(np.nanmedian(sky_map)) if sky_map is not None else _SENTINEL,
        "BGSKYRNG": _fin(np.nanpercentile(sky_map, 95) - np.nanpercentile(sky_map, 5))
                    if sky_map is not None else _SENTINEL,             # sky gradient [DN]
        "BGLANEAM": _fin(np.nanstd(lane_corr)) if lane_corr is not None else _SENTINEL,  # applied stripe amplitude [DN]
        "BGCOVF"  : _fin((weight > 0).mean()) if weight is not None else _SENTINEL,      # coadd model coverage
    }


def read_stats_from_header(path):
    """Recover stats from a previously written corrected exposure."""
    try:
        h = fitsio.read_header(path)
    except Exception:
        return None
    if "BGSTAT" not in h:
        return None
    s = {k: h.get(k, _SENTINEL) for k in _BG_KEYS}
    s["MJD"] = h.get("MJD_OBS", _SENTINEL)
    return s


def plot_run_diagnostics(rows, coadd_id, band_str, outdir, nsig=5.0):
    """
    One 2x3 diagnostic figure per coadd_id and band, plus an outlier list.
    rows: list of dicts with keys path, status, MJD, BG* (from _collect_stats).
    Saves to outdir/plots/<coadd_id>/bgcorr_diag_<band>.png, all_<band>.csv, outliers_<band>.csv
    """
    rows = [r for r in rows if r.get("BGSTAT") is not None]
    if not rows:
        print(f"  [plots] no stats for {coadd_id} {band_str}")
        return

    def col(k):
        a = np.array([r.get(k, _SENTINEL) for r in rows], dtype=float)
        a[a == _SENTINEL] = np.nan
        return a

    stat  = np.array([r["BGSTAT"] for r in rows])
    mjd   = col("MJD")
    x     = mjd if np.isfinite(mjd).sum() > 0.5 * len(mjd) else np.arange(len(rows))
    xlab  = "MJD" if x is mjd else "exposure index"
    wr, mf, sm, sr, la = col("BGWRANGE"), col("BGMASKF"), col("BGSKYMED"), col("BGSKYRNG"), col("BGLANEAM")

    colors = {"OK": "C0", "BIGSKY": "C3", "NOPROJ": "C1"}

    def robust_out(a):
        med = np.nanmedian(a)
        mad = 1.4826 * np.nanmedian(np.abs(a - med))
        return np.isfinite(a) & (np.abs(a - med) > nsig * mad) if mad > 0 else np.zeros_like(a, bool)

    o_wr, o_la, o_mf = robust_out(wr), robust_out(la), robust_out(mf)
    out = o_wr | o_la | o_mf | (stat != "OK")
    cv  = col("BGCOVF")
    reason = np.array(["+".join(n for n, f in (("wrange", o_wr[i]), ("laneamp", o_la[i]), ("maskf", o_mf[i]), ("status", stat[i] != "OK")) if f) for i in range(len(rows))])

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    panels = [
        (ax[0, 0], x, wr, xlab, "work_res range p95-p5 [DN]"),
        (ax[0, 1], x, mf, xlab, "masked fraction of work_res"),
        (ax[0, 2], x, sm, xlab, "sky median [DN]"),
        (ax[1, 0], x, la, xlab, "applied lane correction std [DN]"),
        (ax[1, 1], mf, la, "masked fraction", "applied lane correction std [DN]"),
    ]
    for a, xx, yy, xl, yl in panels:
        for s, c in colors.items():
            m = stat == s
            if m.any():
                a.scatter(xx[m], yy[m], s=3, c=c, alpha=0.5, label=f"{s} ({m.sum()})")
        a.scatter(xx[out], yy[out], s=15, facecolors="none", edgecolors="k", lw=0.6)
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3)
    ax[0, 0].axhline(300, color="r", ls=":", lw=1)          # skip threshold
    ax[0, 0].legend(fontsize=8, markerscale=3)

    ax[1, 2].hist(wr[np.isfinite(wr)], bins=100, log=True, color="C0")
    ax[1, 2].axvline(300, color="r", ls=":", lw=1)
    ax[1, 2].set_xlabel("work_res range [DN]"); ax[1, 2].set_ylabel("N")

    fig.suptitle(f"{coadd_id}  {band_str}   N={len(rows)}   outliers (circled)={out.sum()}")
    plt.tight_layout()

    pdir = os.path.join(outdir, "plots", coadd_id)
    os.makedirs(pdir, exist_ok=True)
    fig.savefig(os.path.join(pdir, f"bgcorr_diag_{band_str}.png"), dpi=120)
    plt.close(fig)

    header = "path,MJD,status,wrange,maskf,skymed,skyrng,laneamp,covf,outlier,reason\n"
    fmt = "{},{:.5f},{},{:.2f},{:.4f},{:.2f},{:.2f},{:.4f},{:.3f},{},{}\n"

    with open(os.path.join(pdir, f"all_{band_str}.csv"), "w") as fa, open(os.path.join(pdir, f"outliers_{band_str}.csv"), "w") as fo:
        fa.write(header); fo.write(header)
        for i in range(len(rows)):
            line = fmt.format(rows[i]["path"], mjd[i], stat[i], wr[i], mf[i], sm[i], sr[i], la[i], cv[i], int(out[i]), reason[i] or "-")
            fa.write(line)
            if out[i]:
                fo.write(line)
    print(f"  [plots] {pdir}  ({out.sum()} outliers)")


# ---------------------------------------------------------------------------
# Integration with unwise_coadd.py (unwise_coadd.py itself is NOT modified).
#
# Pipeline: reproduce unwise_coadd.one_coadd's frame-quality rejection (cheap,
# vectorized table cuts -- qual_frame, bad scans, planets, dtanneal, moon
# masking) to get the same set of frames it would actually use, background/
# lane-correct only those into a mirror of the standard L1b directory layout,
# then have unwise_coadd.one_coadd() build the final coadd from the corrected
# files. unwise_coadd.get_l1b_file is temporarily wrapped (and restored in a
# finally block) so it prefers the corrected copy when one exists, falling
# back to the original raw L1b file otherwise.
# ---------------------------------------------------------------------------

def filter_used_wise_frames(WISE, band, recover_warped=False):
    """
    Reproduce the WISE.use quality cuts from unwise_coadd.one_coadd (qual_frame,
    bad scans, planets, dtanneal, band4 bad-scan range, intmedian, moon masking)
    and cut WISE down to only the frames it would keep. These are vectorized
    table ops over a few thousand rows at most, so this runs in well under a
    second -- background-correcting on this filtered set avoids wasting time on
    exposures unwise_coadd would reject anyway.
    """
    WISE.use = np.ones(len(WISE), bool)
    WISE.use *= (WISE.qual_frame > 0)
    WISE.use *= good_scan_mask(WISE)
    WISE.use *= (WISE.planets == 0)
    if not recover_warped:
        WISE.use *= (WISE.nearby_planets == 0)
    if band in [3, 4]:
        WISE.use *= (WISE.dtanneal > 2000.)
    if band == 4:
        ok = np.array([np.logical_or(s < '03752a', s > '03761b') for s in WISE.scan_id])
        WISE.use *= ok
    WISE.use *= np.isfinite(WISE.intmedian)

    if np.sum(WISE.moon_masked[WISE.use]):
        moon = WISE.moon_masked[WISE.use]
        nomoon = np.logical_not(moon)
        Imoon = np.flatnonzero(WISE.use)[moon]
        nomoonstdevs = WISE.intmed16p[WISE.use][nomoon]
        med = np.median(nomoonstdevs)
        mad = 1.4826 * np.median(np.abs(nomoonstdevs - med))
        moonstdevs = WISE.intmed16p[WISE.use][moon]
        okmoon = (moonstdevs - med) / mad < 5. if mad > 0 else np.ones(len(moonstdevs), bool)
        if not recover_warped:
            WISE.use[Imoon] *= okmoon

    n_before = len(WISE)
    WISE.cut(WISE.use)
    print(f'  [bg-corr] frame rejection: {n_before} -> {len(WISE)} used frames')
    return WISE


@lru_cache(maxsize=1)
def _l1b_dirs():
    return unwise_coadd.get_l1b_dirs(yml=True, verbose=False)

_DL_ROOT = None
def wise_frame_to_l1b_path(wise, band, int_gz=False):
    """Locate the on-disk L1b intensity file for one WISE frame (raw or 'missing' dir)."""
    phase = phase_from_scanid(wise.scan_id)
    wdirs = _l1b_dirs()
    dl_dir = os.path.join(_DL_ROOT, 'merge_p1bm_frm') if _DL_ROOT else None
    for wdir in (wdirs.get(phase), wdirs.get('missing'), dl_dir):
        if wdir is None:
            continue
        intfn = unwise_coadd.get_l1b_file(wdir, wise.scan_id, wise.frame_num, band, int_gz=int_gz)
        found = existing_path(intfn)
        if found:
            return found
    return None


def correct_exposures_for_tile(WISE, band_num, model_dir, release, atlas_path,
                                corr_outdir, coadd_id=None, nthreads=16,
                                force=False, int_gz=False, use_zp_meta=False):
    """
    Background/star/lane-correct the L1b exposures in `WISE` (already cut down
    to the frames unwise_coadd would use -- see filter_used_wise_frames),
    writing corrected int/unc/msk triplets into corr_outdir using the same
    directory layout as unwise_utils.get_l1b_file.

    use_zp_meta: see init_worker -- default False uses zp_lookup.ZPLookUp,
    matching unwise_coadd.py's own default zeropoint source.

    Returns (results, missing): results is one dict per exposure found on disk,
    and missing is the {(scan_id, frame_num)} set of WISE rows with no L1b
    file on disk at all, so callers can exclude those from the coadd too.
    """
    tag = coadd_id or 'tile'
    paths = [wise_frame_to_l1b_path(w, band_num, int_gz=int_gz) for w in WISE]
    missing = {(w.scan_id.strip(), int(w.frame_num)) for w, p in zip(WISE, paths) if p is None}
    valid_paths = [p for p in paths if p is not None]
    if missing:
        print(f'  [bg-corr] {len(missing)} of {len(paths)} used frames missing on disk')
    if not valid_paths:
        print(f'  [bg-corr] no on-disk exposures to correct for {tag} w{band_num}')
        return [], missing

    atlas_data = fitsio.read(atlas_path)

    results = [None] * len(valid_paths)
    with Pool(processes=nthreads, initializer=init_worker,
              initargs=(atlas_data, model_dir, band_num, corr_outdir, release, force, use_zp_meta)) as pool:
        for idx, status, stats in pool.imap_unordered(process_one_exposure, list(enumerate(valid_paths))):
            results[idx] = {'path': valid_paths[idx], 'status': status, **(stats or {})}

    n_ok = sum(1 for r in results if r['status'] in ('success', 'pre-existing'))
    print(f'  [bg-corr] corrected {n_ok}/{len(results)} exposures for {tag} w{band_num}')
    return results, missing


def _wrap_get_l1b_file(orig_get_l1b_file, corr_outdir, bad=frozenset(), fallback_log=None):
    """
    Point unwise_coadd at the corrected copy of each exposure.
    Frames in `bad` (missing on disk or failed correction) get a path that never
    exists, so one_coadd marks them 'Not found' -- exactly how the standalone run
    treats missing files -- instead of them being cut from the frame table.
    """
    def wrapped(basedir, scanid, frame, band, int_gz=False):
        if (str(scanid).strip(), int(frame)) in bad:
            corr = orig_get_l1b_file(corr_outdir, scanid, frame, band, int_gz=int_gz)
            return corr + '.excluded'
        # Check corrected copy
        corr = orig_get_l1b_file(corr_outdir, scanid, frame, band, int_gz=int_gz)
        found = existing_path(corr)
        if found:
            return found
        # Fall back to raw (including IRSA-downloaded files in merge_p1bm_frm)
        if fallback_log is not None:
            fallback_log.append((scanid, frame))
        return orig_get_l1b_file(basedir, scanid, frame, band, int_gz=int_gz)
    return wrapped

def _init_unwise_coadd_globals(int_gz=False, use_zp_meta=False):
    """Set unwise_coadd's module globals, as its own main() does. Must run
    BEFORE any multiproc pool is created, because workers are forked with a
    copy of these globals."""
    if unwise_coadd.logger is None:
        logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
        unwise_coadd.logger = logging.getLogger('unwise_coadd')
    unwise_coadd.int_gz = int_gz
    # Same flag/default as unwise_coadd.py's own --use_zp_meta: False uses
    # zp_lookup.ZPLookUp; True uses the raw MAGZP header card. The L1b
    # stripe correction uses this same setting (see correct_exposures_for_tile),
    # so the correction's photometric scaling matches the coadd's own.
    unwise_coadd.use_zp_meta = use_zp_meta
    if unwise_coadd.compare_moon_all is None:
        unwise_coadd.compare_moon_all = False
        

def run_corrected_coadd(coadd_id, band_num, model_dir, release, atlas_path,
                         corr_outdir, final_outdir, W=2048, H=2048, pixscale=2.75,
                         nthreads=16, force_bg=False, force=False, int_gz=False,
                         make_plots=True, one_coadd_kwargs=None,
                         save_corrected_images=False, use_zp_meta=False):
    """
    End-to-end: select the L1b exposures unwise_coadd would use for one coadd
    tile, background-correct only those, then build the final coadd from the
    corrected frames by calling unwise_coadd.one_coadd(). unwise_coadd.py is
    not modified.

    corr_outdir: where corrected int/unc/msk triplets are written. If None,
        a private temporary directory is created and used instead (and
        always cleaned up afterwards, regardless of save_corrected_images,
        since the caller never named a path it could reuse).
    save_corrected_images: if True, leave the corrected L1b files in
        corr_outdir after the run (useful to reuse/cache them across
        overlapping tiles, or for inspection/debugging). If False, the
        corrected files are deleted once the coadd has been built -- but
        only when corr_outdir was auto-created here; an explicitly-passed
        corr_outdir is never auto-deleted, since it may be a shared cache
        directory used by other runs/tiles.
    use_zp_meta: same flag/semantics as unwise_coadd.py's own --use_zp_meta.
        Default False: both the stripe correction and the final coadd get
        their per-frame zeropoint from zp_lookup.ZPLookUp(band, poly=True).
        True: both use the raw MAGZP header card instead.
    """
    tile = unwise_coadd.get_atlas_tiles(0., 360., -90., 90., coadd_id=coadd_id)
    assert len(tile) == 1
    tile = tile[0]

    _init_unwise_coadd_globals(int_gz, use_zp_meta)

    # Absolute paths, so the chdir below doesn't break any output location
    global _DL_ROOT
    owns_corr_outdir = corr_outdir is None
    if owns_corr_outdir:
        corr_outdir = tempfile.mkdtemp(prefix='unwise_bgcorr_')
    corr_outdir  = os.path.abspath(corr_outdir)
    final_outdir = os.path.abspath(final_outdir)
    _DL_ROOT     = corr_outdir          # downloads go to corr_outdir/merge_p1bm_frm
    os.makedirs(_DL_ROOT, exist_ok=True)

    recover_warped = (one_coadd_kwargs or {}).get('recover_warped', False)

    # Frames to CORRECT: a separately built, quality-filtered table.
    WISE_corr = unwise_coadd.get_wise_frames(tile.ra, tile.dec, band_num)
    WISE_corr = filter_used_wise_frames(WISE_corr, band_num, recover_warped=recover_warped)

    results, missing = correct_exposures_for_tile(
        WISE_corr, band_num, model_dir, release, atlas_path, corr_outdir,
        coadd_id=coadd_id, nthreads=nthreads, force=force_bg, int_gz=int_gz,
        use_zp_meta=use_zp_meta)

    if make_plots and results:
        plot_run_diagnostics(results, coadd_id, 'w%i' % band_num, final_outdir)

    # Only failed corrections go in bad; missing frames may be downloadable
    # They are NOT removed from the table; the wrapper makes them 'Not found'.
    bad = set()
    for r in results:
        if r['status'] in ('success', 'pre-existing') or r['status'].startswith('skipped_'):
            continue
        base = os.path.basename(r['path']).replace('.gz', '')
        bad.add((base[:6], int(base[6:9])))
    print(f'  [bg-corr] {len(bad)} frame(s) without a corrected copy will be treated as not found')

    # Frames for the COADD: a fresh, untouched table, exactly as the standalone
    # unwise_coadd.py builds it (default 1.7 deg margin, no pre-cuts).
    WISE = unwise_coadd.get_wise_frames(tile.ra, tile.dec, band_num)

    medfilt = _resolve_medfilt(None, band_num)

    kwargs = dict(
        ps=None, wishlist=False, outdir=final_outdir, mp1=None, mp2=None,
        do_cube=False, plots2=False, frame0=0, nframes=0, nframes_random=0,
        force=force, medfilt=medfilt, maxmem=0, do_dsky=False, checkmd5=False,
        bgmatch=False, center=False, minmax=False, rchi_fraction=0.01, do_cube1=False,
        epoch=None, before=100000.0, after=None, recover_warped=recover_warped, do_rebin=True,
        try_download=False, hi_lo_rej=False, output_masks=True,
    )
    if one_coadd_kwargs:
        kwargs.update(one_coadd_kwargs)

    fallback_log = []
    orig_get_l1b_file = unwise_coadd.get_l1b_file
    unwise_coadd.get_l1b_file = _wrap_get_l1b_file(orig_get_l1b_file, corr_outdir, bad=frozenset(bad), fallback_log=fallback_log)
    cwd = os.getcwd()
    os.chdir(_DL_ROOT)                  # one_coadd's hardcoded 'merge_p1bm_frm' resolves here
    try:
        rtn = unwise_coadd.one_coadd(tile, band_num, W, H, pixscale, WISE, **kwargs)
    finally:
        os.chdir(cwd)
        unwise_coadd.get_l1b_file = orig_get_l1b_file
        if owns_corr_outdir and not save_corrected_images:
            shutil.rmtree(corr_outdir, ignore_errors=True)

    if fallback_log:
        print(f'  [bg-corr] WARNING: {len(fallback_log)} frame(s) fell back to raw '
              f'L1b data (not background-corrected): {fallback_log}')

    return rtn


def main():
    parser = argparse.ArgumentParser(
        description='Background/lane-correct L1b exposures for a unWISE tile, '
                     'then build the coadd from the corrected frames.')
    parser.add_argument('--tile', required=True, help='coadd_id, e.g. 2709p666')
    parser.add_argument('--band', type=int, required=True)
    parser.add_argument('--model-dir', required=True, help='crowdsource star model directory (*.mod.fits)')
    parser.add_argument('--release', required=True, help='unwise release tag used to find -msk.fits.gz artifact masks')
    parser.add_argument('--atlas', required=True, help='atlas FITS table with CRVAL and COADD_ID columns')
    parser.add_argument('--corr-outdir', default=None,
                      help=('directory to write background-corrected L1b files. '
                            'Default: a private temporary directory, which is deleted '
                            'after the coadd is built unless --save-corrected-images is given.'))
    parser.add_argument('--save-corrected-images', dest='save_corrected_images', action='store_true',
                      default=False,
                      help=('keep the background-corrected L1b files in --corr-outdir after the run '
                            '(e.g. to reuse/cache them across overlapping tiles, or for inspection). '
                            'Default: delete them once the coadd has been built. Has no effect -- the '
                            'files are never auto-deleted -- if --corr-outdir was explicitly given, '
                            'since that directory may be a shared cache used by other runs.'))
    parser.add_argument('--outdir', required=True, help='final coadd output directory')
    parser.add_argument('--nthreads', type=int, default=16)
    parser.add_argument('--force-bg', action='store_true', help='re-run correction even if cached output exists')
    parser.add_argument('--no-plots', dest='no_plots', action='store_true', help='skip writing diagnostic plots/CSVs')

    # Everything below is copied verbatim (flag string, dest, type/action,
    # default, help) from unwise_coadd.py's own argparse, so it passes
    # through to unwise_coadd.one_coadd() with identical names/definitions.
    parser.add_argument('-w', dest='wishlist', action='store_true',
                      default=False, help='Print needed frames and exit?')
    parser.add_argument('--plots2', dest='plots2', action='store_true',
                      default=False)
    parser.add_argument('--cube', dest='cube', action='store_true',
                      default=False, help='Save & write out image cube')
    parser.add_argument('--cube1', dest='cube1', action='store_true',
                      default=False, help='Save & write out image cube for round 1')
    parser.add_argument('--frame0', dest='frame0', default=0, type=int,
                      help='Only use a subset of the frames: starting with frame0')
    parser.add_argument('--nframes', dest='nframes', default=0, type=int,
                      help='Only use a subset of the frames: number nframes')
    parser.add_argument('--nframes-random', dest='nframes_random', default=0, type=int,
                      help='Only use a RANDOM subset of the frames: number nframes')
    parser.add_argument('--medfilt', dest='medfilt', type=int, default=None,
                      help=('Median filter with a box twice this size (+1),'+
                            ' to remove varying background.  Default: none for W1,W2; 50 for W3,W4.'))
    parser.add_argument('--maxmem', dest='maxmem', type=float, default=0,
                      help='Quit if predicted memory usage > n GB')
    parser.add_argument('--dsky', dest='dsky', action='store_true',
                      default=False,
                      help='Do background-matching by matching medians '
                      '(to first-round coadd)')
    parser.add_argument('--bgmatch', dest='bgmatch', action='store_true',
                      default=False,
                      help='Do background-matching by matching medians '
                      '(when accumulating first-round coadd)')
    parser.add_argument('--center', dest='center', action='store_true',
                      default=False,
                      help='Read frames in order of distance from center; for debugging.')
    parser.add_argument('--minmax', action='store_true',
                      help='Record the minimum and maximum values encountered during coadd?')
    parser.add_argument('--rchi-fraction', dest='rchi_fraction', type=float,
                      default=0.01, help='Fraction of outlier pixels to reject frame')
    parser.add_argument('--epoch', type=int, help='Keep only input frames in the given epoch, zero-indexed')
    parser.add_argument('--before', type=float, help='Keep only input frames before the given MJD',
                      default=100000.0)
    parser.add_argument('--after',  type=float, help='Keep only input frames after the given MJD')
    parser.add_argument('--recover_warped', dest='recover_warped', action='store_true', default=False,
                      help='Attempt to recover Moon-contaminated exposures?')
    parser.add_argument('--no_warp_rebin', dest='do_rebin', action='store_false', default=True,
                      help='Turn of rebinning when fitting per-quadrant polynomial warps.')
    parser.add_argument('--no_irsa_dl', dest='try_download', action='store_false', default=True,
                      help='Do not attempt to download missing L1b files on the fly from IRSA.')
    parser.add_argument('--hi_lo_rej', dest='hi_lo_rej', action='store_true', default=False,
                      help='Include a min/max rejection stpe during first round coaddition.')
    parser.add_argument('--no_output_masks', dest='output_masks', action='store_false', default=True,
                      help='Turn off writing of per-exposure mask outputs.')
    parser.add_argument('--threads', dest='threads', type=int, default=None, help='Multiproc')
    parser.add_argument('--threads1', dest='threads1', type=int, default=None,
                      help='Multithreading during round 1')
    parser.add_argument('--force', dest='force', action='store_true',
                      default=False, help='Run even if output file already exists?')
    parser.add_argument('--use_zp_meta', dest='use_zp_meta', action='store_true', default=False,
                      help=('Should coadd use MAGZP metadata for zero points? (same flag as '
                            'unwise_coadd.py; also controls the zeropoint used by the stripe '
                            'correction itself, so both stay consistent). Default: use '
                            'zp_lookup.ZPLookUp.'))
    args = parser.parse_args()

    # unwise_coadd.py's own resolution of --medfilt: None -> 50 for W3,W4 else 0.
    medfilt = _resolve_medfilt(args.medfilt, args.band)

    # Must happen before mp1/mp2 pools below are created (forked workers get
    # a copy of unwise_coadd's globals at that point).
    _init_unwise_coadd_globals(use_zp_meta=args.use_zp_meta)

    # mirrors unwise_coadd.py main()'s own --threads/--threads1 -> mp1/mp2 logic
    from astrometry.util.multiproc import multiproc
    mp2 = multiproc(args.threads) if args.threads else multiproc()
    mp1 = mp2 if args.threads1 is None else multiproc(args.threads1)

    one_coadd_kwargs = dict(
        wishlist=args.wishlist, do_cube=args.cube, plots2=args.plots2,
        frame0=args.frame0, nframes=args.nframes, nframes_random=args.nframes_random,
        medfilt=medfilt, maxmem=args.maxmem, do_dsky=args.dsky, bgmatch=args.bgmatch,
        center=args.center, minmax=args.minmax, rchi_fraction=args.rchi_fraction,
        do_cube1=args.cube1, epoch=args.epoch, before=args.before, after=args.after,
        recover_warped=args.recover_warped, do_rebin=args.do_rebin,
        try_download=args.try_download, hi_lo_rej=args.hi_lo_rej,
        output_masks=args.output_masks, mp1=mp1, mp2=mp2,
    )

    rtn = run_corrected_coadd(
        args.tile, args.band, args.model_dir, args.release, args.atlas,
        args.corr_outdir, args.outdir, nthreads=args.nthreads,
        force_bg=args.force_bg, force=args.force,
        make_plots=not args.no_plots, one_coadd_kwargs=one_coadd_kwargs,
        save_corrected_images=args.save_corrected_images,
        use_zp_meta=args.use_zp_meta,
    )
    return rtn


if __name__ == '__main__':
    sys.exit(main() or 0)
