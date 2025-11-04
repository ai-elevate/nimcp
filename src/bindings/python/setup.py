"""
NIMCP Python Bindings Setup v2.7.0
===================================

WHAT: Build configuration for NIMCP Python extension
WHY:  Enable pip/setuptools installation of Python bindings
HOW:  Define C extension module with proper includes/libraries

Installation:
    python setup.py build
    python setup.py install

Or:
    pip install .

Requirements:
    - Python 3.7+
    - NIMCP library (libnimcp.so / libnimcp.dylib / nimcp.dll)
    - C compiler

Usage after installation:
    import nimcp
    brain = nimcp.Brain("test", nimcp.BRAIN_SMALL, nimcp.TASK_CLASSIFICATION, 10, 5)
"""

from setuptools import setup, Extension
import os
import sys

# Detect NIMCP library location
nimcp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
nimcp_include = os.path.join(nimcp_root, 'src', 'include')
nimcp_lib = os.path.join(nimcp_root, 'bin')

# Define extension module
nimcp_extension = Extension(
    'nimcp',
    sources=['nimcp_python.c'],  # Enhanced v2.7.0 implementation
    include_dirs=[nimcp_include],
    library_dirs=[nimcp_lib],
    libraries=['nimcp'],
    extra_compile_args=['-std=c11', '-Wall', '-Wextra'],
    runtime_library_dirs=[nimcp_lib] if sys.platform != 'win32' else None
)

setup(
    name='nimcp',
    version='2.7.0',
    description='Python bindings for NIMCP - Neural Interface Message Communication Protocol',
    long_description='NIMCP Python bindings with v2.7.0 enhancements: batch processing, async inference, checkpointing, SIMD operations',
    author='NIMCP Development Team',
    license='MIT',
    ext_modules=[nimcp_extension],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='neural-network machine-learning ai cognitive-computing',
)
