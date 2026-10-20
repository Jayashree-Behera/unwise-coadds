#!/usr/bin/env python
'''
Compare two coadd output trees for exact equality.

Pixel data must match bit-for-bit; that is the whole claim being tested.
Headers are reported separately because FITS writers stamp creation dates
and software versions, so a plain byte comparison of the files would fail
for reasons that have nothing to do with the numbers.

    python compare_outputs.py <reference_dir> <test_dir>
'''
from __future__ import print_function
import os
import sys

import numpy as np

try:
    import fitsio
except ImportError:
    fitsio = None
    from astropy.io import fits as _afits

# Header keys expected to differ between two runs of the same code.
VOLATILE = set(['DATE', 'DATE-OBS', 'COMMENT', 'HISTORY', 'CHECKSUM',
                'DATASUM', 'SOFTWARE', 'VERSION', 'UNWVER', 'WALLTIME',
                'CPUTIME'])


class _AstropyFITS(object):
    '''Minimal stand-in for fitsio.FITS, so this runs anywhere astropy is
    available as well as on systems carrying fitsio.'''
    def __init__(self, fn):
        self._h = _afits.open(fn, memmap=False)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._h.close()

    def __len__(self):
        return len(self._h)

    def __getitem__(self, i):
        hdu = self._h[i]
        return _AstropyHDU(hdu)


class _AstropyHDU(object):
    def __init__(self, hdu):
        self._hdu = hdu

    def read(self):
        return self._hdu.data

    def read_header(self):
        return self._hdu.header


def open_fits(fn):
    if fitsio is not None:
        return fitsio.FITS(fn)
    return _AstropyFITS(fn)


def fits_files(root):
    out = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.endswith(('.fits', '.fits.gz', '.fit')):
                full = os.path.join(dirpath, n)
                out[os.path.relpath(full, root)] = full
    return out


def compare_file(a, b):
    '''Returns (data_ok, n_hdu, [header differences]).'''
    hdrdiff = []
    with open_fits(a) as fa, open_fits(b) as fb:
        # Capture this inside the block: the handles are closed on exit and
        # cannot be queried afterwards.
        nhdu = len(fa)
        if nhdu != len(fb):
            return False, 0, ['HDU count %i vs %i' % (nhdu, len(fb))]
        for i in range(nhdu):
            da, db = fa[i].read(), fb[i].read()
            if da is None and db is None:
                pass
            elif da is None or db is None:
                return False, nhdu, hdrdiff + ['HDU %i: one side empty' % i]
            elif da.dtype != db.dtype or da.shape != db.shape:
                return False, nhdu, hdrdiff + [
                    'HDU %i: %s%s vs %s%s' % (i, da.dtype, da.shape,
                                              db.dtype, db.shape)]
            # Exact, including NaN placement: array_equal treats NaN as
            # unequal, so compare the bit patterns instead.
            elif da.dtype.kind == 'f':
                if da.view(np.uint8).tobytes() != db.view(np.uint8).tobytes():
                    n = int(np.sum(~((da == db) | (np.isnan(da) & np.isnan(db)))))
                    return False, nhdu, hdrdiff + [
                        'HDU %i: %i differing pixels of %i'
                        % (i, n, da.size)]
            elif not np.array_equal(da, db):
                n = int(np.sum(da != db))
                return False, nhdu, hdrdiff + [
                    'HDU %i: %i differing values of %i' % (i, n, da.size)]

            ha, hb = fa[i].read_header(), fb[i].read_header()
            for k in set(list(ha.keys()) + list(hb.keys())):
                if k.upper() in VOLATILE:
                    continue
                va, vb = ha.get(k), hb.get(k)
                if va != vb:
                    hdrdiff.append('HDU %i %s: %r vs %r' % (i, k, va, vb))
    return True, nhdu, hdrdiff


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ref, test = sys.argv[1], sys.argv[2]
    fa, fb = fits_files(ref), fits_files(test)

    only_ref = sorted(set(fa) - set(fb))
    only_test = sorted(set(fb) - set(fa))
    common = sorted(set(fa) & set(fb))

    for n in only_ref:
        print('MISSING from test : %s' % n)
    for n in only_test:
        print('EXTRA   in   test : %s' % n)

    nbad = 0
    nhdr = 0
    for n in common:
        ok, _, hdr = compare_file(fa[n], fb[n])
        if not ok:
            nbad += 1
            print('DATA DIFFERS      : %s' % n)
            for d in hdr:
                print('                    %s' % d)
        elif hdr:
            nhdr += 1
            print('headers differ    : %s' % n)
            for d in hdr[:5]:
                print('                    %s' % d)

    print('')
    print('%i files compared, %i data mismatches, %i header-only differences'
          % (len(common), nbad, nhdr))
    print('%i missing, %i extra' % (len(only_ref), len(only_test)))
    bad = nbad or only_ref or only_test
    print('RESULT: %s' % ('MISMATCH' if bad else 'BIT-IDENTICAL'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
