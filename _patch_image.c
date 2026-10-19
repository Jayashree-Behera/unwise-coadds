/*
 * C implementation of patch_image (see fast_patch.py for the Python
 * reference and for why this is worth doing).
 *
 * patch_image is an iterative flood fill: on each pass every still-masked
 * pixel averages its good 4-neighbours, becomes good if it had any, and the
 * pass repeats until nothing is left. Expressed in numpy each pass allocates
 * and traverses several temporary arrays per neighbour offset, so a frame
 * with large masked blobs -- which needs one pass per pixel of blob radius --
 * spends most of its time in allocation and memory traffic rather than
 * arithmetic. In C the whole pass is a single loop over a shrinking index
 * list with no temporaries beyond two small scratch arrays.
 *
 * The result is bit-for-bit identical to the numpy version, which requires:
 *
 *   - psum is a float (single precision) accumulator, added to in dxdy order,
 *     so the rounding sequence matches numpy's per-offset float32 adds.
 *   - a neighbour contributes `v * (mask ? 1.0f : 0.0f)`, not a branch on the
 *     mask, so that the signed zero behaviour of `v * False` matches.
 *   - a non-finite neighbour is skipped entirely before the mask is even
 *     consulted, matching numpy's `ok[ok] *= finite`.
 *   - the final value is (float)((double)psum / (double)max(pn,1)): numpy
 *     promotes float32/int64 to float64 and `.astype` rounds back down.
 *   - img is written for every pending index, including those with pn == 0,
 *     which is what replaces a NaN at an unreachable pixel with 0.0.
 *   - all neighbour reads in a pass see the values from the start of the
 *     pass, so img/mask are only written after the whole pass is scanned.
 *
 * Arbitrary 2-D strides are supported because round 1 patches a
 * non-contiguous view into the full frame.
 *
 * Build:  python setup_patch.py build_ext --inplace
 * If it is not built, fast_patch.py falls back to the numpy implementation.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static const int DEF_DX[4] = {-1, 1, 0, 0};
static const int DEF_DY[4] = {0, 0, -1, 1};

static inline float *fat(char *base, npy_intp sy, npy_intp sx,
                         npy_intp y, npy_intp x)
{
    return (float *)(base + y * sy + x * sx);
}

static inline npy_bool *bat(char *base, npy_intp sy, npy_intp sx,
                            npy_intp y, npy_intp x)
{
    return (npy_bool *)(base + y * sy + x * sx);
}

static PyObject *
patch_image_c(PyObject *self, PyObject *args)
{
    PyArrayObject *imgo, *masko, *reqo;
    PyObject *reqarg;

    if (!PyArg_ParseTuple(args, "O!O!O", &PyArray_Type, &imgo,
                          &PyArray_Type, &masko, &reqarg))
        return NULL;

    if (PyArray_TYPE(imgo) != NPY_FLOAT32 || PyArray_NDIM(imgo) != 2) {
        PyErr_SetString(PyExc_TypeError, "img must be 2-D float32");
        return NULL;
    }
    if (PyArray_TYPE(masko) != NPY_BOOL || PyArray_NDIM(masko) != 2) {
        PyErr_SetString(PyExc_TypeError, "mask must be 2-D bool");
        return NULL;
    }

    reqo = NULL;
    if (reqarg != Py_None) {
        if (!PyArray_Check(reqarg)) {
            PyErr_SetString(PyExc_TypeError, "required must be an array or None");
            return NULL;
        }
        reqo = (PyArrayObject *)reqarg;
        if (PyArray_TYPE(reqo) != NPY_BOOL || PyArray_NDIM(reqo) != 2) {
            PyErr_SetString(PyExc_TypeError, "required must be 2-D bool");
            return NULL;
        }
    }

    const npy_intp h = PyArray_DIM(imgo, 0);
    const npy_intp w = PyArray_DIM(imgo, 1);
    if (PyArray_DIM(masko, 0) != h || PyArray_DIM(masko, 1) != w) {
        PyErr_SetString(PyExc_ValueError, "img and mask shapes differ");
        return NULL;
    }
    if (reqo && (PyArray_DIM(reqo, 0) != h || PyArray_DIM(reqo, 1) != w)) {
        PyErr_SetString(PyExc_ValueError, "img and required shapes differ");
        return NULL;
    }

    char *imgb = PyArray_BYTES(imgo);
    char *maskb = PyArray_BYTES(masko);
    const npy_intp isy = PyArray_STRIDE(imgo, 0), isx = PyArray_STRIDE(imgo, 1);
    const npy_intp msy = PyArray_STRIDE(masko, 0), msx = PyArray_STRIDE(masko, 1);

    npy_intp *ys = NULL, *xs = NULL;
    float *psum = NULL;
    npy_intp *pn = NULL;
    npy_intp n = 0;
    int result = 1;

    /* Build the initial pending list, in ascending flat-index order so that
     * the traversal order matches np.flatnonzero. */
    {
        npy_intp cap = 0, y, x;
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                if (*bat(maskb, msy, msx, y, x))
                    continue;
                if (reqo && !*bat(PyArray_BYTES(reqo), PyArray_STRIDE(reqo, 0),
                                  PyArray_STRIDE(reqo, 1), y, x))
                    continue;
                cap++;
            }
        }
        if (cap == 0)
            Py_RETURN_TRUE;

        ys = (npy_intp *)malloc(sizeof(npy_intp) * cap);
        xs = (npy_intp *)malloc(sizeof(npy_intp) * cap);
        psum = (float *)malloc(sizeof(float) * cap);
        pn = (npy_intp *)malloc(sizeof(npy_intp) * cap);
        if (!ys || !xs || !psum || !pn) {
            free(ys); free(xs); free(psum); free(pn);
            return PyErr_NoMemory();
        }
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                if (*bat(maskb, msy, msx, y, x))
                    continue;
                if (reqo && !*bat(PyArray_BYTES(reqo), PyArray_STRIDE(reqo, 0),
                                  PyArray_STRIDE(reqo, 1), y, x))
                    continue;
                ys[n] = y;
                xs[n] = x;
                n++;
            }
        }
    }

    Py_BEGIN_ALLOW_THREADS

    {
        npy_intp nlast = -1;
        while (n > 0) {
            npy_intp k, kept;
            if (n == nlast) {
                result = 0;
                break;
            }
            nlast = n;

            for (k = 0; k < n; k++) {
                const npy_intp y = ys[k], x = xs[k];
                float acc = 0.0f;
                npy_intp cnt = 0;
                int d;
                for (d = 0; d < 4; d++) {
                    const npy_intp ny = y + DEF_DY[d];
                    const npy_intp nx = x + DEF_DX[d];
                    float v, mf;
                    if (ny < 0 || ny >= h || nx < 0 || nx >= w)
                        continue;
                    v = *fat(imgb, isy, isx, ny, nx);
                    if (!isfinite(v))
                        continue;
                    mf = *bat(maskb, msy, msx, ny, nx) ? 1.0f : 0.0f;
                    acc = acc + v * mf;
                    cnt += (npy_intp)(mf != 0.0f);
                }
                psum[k] = acc;
                pn[k] = cnt;
            }

            /* Writes happen only after the whole pass, so every neighbour
             * read above saw the start-of-pass state, as in numpy. */
            for (k = 0; k < n; k++) {
                const npy_intp den = pn[k] > 1 ? pn[k] : 1;
                *fat(imgb, isy, isx, ys[k], xs[k]) =
                    (float)((double)psum[k] / (double)den);
                *bat(maskb, msy, msx, ys[k], xs[k]) =
                    (npy_bool)(pn[k] > 0);
            }

            kept = 0;
            for (k = 0; k < n; k++) {
                if (pn[k] > 0)
                    continue;
                ys[kept] = ys[k];
                xs[kept] = xs[k];
                kept++;
            }
            n = kept;
        }
    }

    Py_END_ALLOW_THREADS

    free(ys);
    free(xs);
    free(psum);
    free(pn);

    if (result)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyMethodDef methods[] = {
    {"patch_image_c", patch_image_c, METH_VARARGS,
     "patch_image(img float32[h,w], mask bool[h,w], required bool[h,w] or None)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_patch_image", NULL, -1, methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__patch_image(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
