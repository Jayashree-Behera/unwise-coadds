"""
Builds the optional C accelerator for patch_image.

    python setup_patch.py build_ext --inplace

fast_patch.py imports the result if it is present and silently uses its numpy
implementation if it is not, so building this is optional and the pipeline
produces identical output either way.
"""
import numpy as np
from setuptools import setup, Extension

setup(
    name='unwise_patch_image',
    ext_modules=[
        Extension('_patch_image',
                  sources=['_patch_image.c'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=['-O2', '-fno-fast-math']),
    ],
)
