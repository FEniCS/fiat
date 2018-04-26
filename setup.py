#!/usr/bin/env python

import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info < (3, 0):
    print("Python 3.0 or higher required, please upgrade.")
    sys.exit(1)

version = "2018.1.0"

url = "https://bitbucket.org/fenics-project/fiat/"
tarball = None
if 'dev' not in version:
    tarball = url + "downloads/fenics-fiat-%s.tar.gz" % version

setup(name="fenics-fiat",
      description="FInite element Automatic Tabulator",
      version=version,
      author="Robert C. Kirby et al.",
      author_email="fenics-dev@googlegroups.com",
      url=url,
      download_url=tarball,
      license="LGPL v3 or later",
      packages=["FIAT"],
      install_requires=["numpy", "sympy"])
