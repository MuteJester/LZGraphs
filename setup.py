"""
Build script for optional C extensions.

The _fast_walk extension accelerates LZGraph.simulate() by ~50-100x.
If compilation fails (no C compiler), the package still installs and
falls back to the pure-Python implementation automatically.
"""

import os
import sys
from setuptools import setup, Extension

# Ensure setuptools can resolve the dynamic version (attr = "LZGraphs.__version__")
# when running in an isolated build environment where src/ isn't on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

ext_modules = [
    Extension(
        "LZGraphs._fast_walk",
        sources=[os.path.join("src", "LZGraphs", "_fast_walk.c")],
        # No external library dependencies — pure C + Python.h
    ),
]


def run_setup(extensions):
    setup(ext_modules=extensions)


try:
    run_setup(ext_modules)
except Exception:
    print(
        "\n"
        "WARNING: Failed to compile C extension _fast_walk.\n"
        "         LZGraphs will use the pure-Python fallback for simulate().\n"
        "         This is fine — the package works without it, just slower.\n"
        "\n"
    )
    run_setup([])
