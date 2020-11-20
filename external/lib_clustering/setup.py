from distutils.core import setup
from Cython.Build import cythonize
import os
import numpy


setup(
    ext_modules = cythonize("clustering.pyx", annotate=True),
    name="lib_clustering",
    include_dirs=[numpy.get_include()],
)

