"""Build script for LZGraphs with C extension.

Compiles the C-LZGraph library and Python bindings into a single
shared library that is imported as LZGraphs._clzgraph.
"""

import os
import sys
import glob
from setuptools import setup, Extension

HERE = os.path.dirname(os.path.abspath(__file__))

# Ensure setuptools can resolve dynamic version
sys.path.insert(0, os.path.join(HERE, "src"))

# Change to setup.py directory so all paths are relative
os.chdir(HERE)

# Collect all C library source files (relative to HERE)
lib_sources = sorted(glob.glob(os.path.join("lib", "**", "*.c"), recursive=True))

# Platform-specific compile flags
if sys.platform == 'win32':
    extra_compile = ["/O2", "/W3"]
    extra_link = []
    macros = []
else:
    extra_compile = ["-O2", "-std=c11", "-Wno-unused-function"]
    extra_link = ["-lm"]
    macros = [("_POSIX_C_SOURCE", "200809L")]

ext = Extension(
    "LZGraphs._clzgraph",
    sources=[
        os.path.join("src", "LZGraphs", "_clzgraph.c"),
    ] + lib_sources,
    include_dirs=["include"],
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
    define_macros=macros,
)

setup(ext_modules=[ext])
