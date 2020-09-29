#!/usr/bin/env python

from setuptools import setup
import sys

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)


setup(name="fenics-fiat",
      description="FInite element Automatic Tabulator",
      author="Robert C. Kirby et al.",
      author_email="fenics-dev@googlegroups.com",
      setup_requires=["setuptools_scm"],
      use_scm_version={"parentdir_prefix_version": "fiat-"},
      url="https://github.com/FEniCS/fiat/",
      license="LGPL v3 or later",
      packages=["FIAT"],
      install_requires=["numpy", "sympy"])
