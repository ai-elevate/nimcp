"""
NIMCP Python Bindings Setup
Uses the unified nimcp.h API
"""

from setuptools import setup, Extension
import os

# Get the path to the build directory
build_dir = os.path.join(os.path.dirname(__file__), '../../../build/src/lib')

nimcp_module = Extension(
    'nimcp',
    sources=['nimcp_py.c'],
    include_dirs=['../../include'],
    library_dirs=[build_dir],
    libraries=['nimcp_core'],
    runtime_library_dirs=[build_dir],
    extra_compile_args=['-std=c11'],
)

setup(
    name='nimcp',
    version='2.6.1',
    description='NIMCP - Neural Interface Message Communication Protocol',
    ext_modules=[nimcp_module],
    python_requires='>=3.6',
)
