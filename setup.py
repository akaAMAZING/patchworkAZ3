"""Build Cython extensions for patchwork. Run: python setup.py build_ext --inplace"""
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    import sys
    print("Cython is required to build the extension. Install it for this Python:", file=sys.stderr)
    print("  ", sys.executable, "-m pip install Cython", file=sys.stderr)
    sys.exit(1)

import numpy as np

ext = Extension(
    "src.mcts.packing_heuristic_cy",
    sources=["src/mcts/packing_heuristic_cy.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="patchwork-mcts",
    ext_modules=cythonize(ext, compiler_directives={"language_level": "3"}),
)
