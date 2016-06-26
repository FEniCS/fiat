#!/usr/bin/env python

import re
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info < (2, 7):
    print("Python 2.7 or higher required, please upgrade.")
    sys.exit(1)

version = re.findall('__version__ = "(.*)"',
                     open('FIAT/__init__.py', 'r').read())[0]

url = "https://bitbucket.org/fenics-project/fiat/"
tarball = None
if 'dev' not in version:
    tarball = url + "downloads/fiat-%s.tar.gz" % version

setup(name="FIAT",
      description="FInite element Automatic Tabulator",
      version=version,
      author="Robert C. Kirby et al.",
      author_email="fenics-dev@googlegroups.com",
      url=url,
      download_url=tarball,
      license="LGPL v3 or later",
      packages=["FIAT"],
      install_requires=["numpy", "sympy", "six"])
