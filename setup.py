#!/usr/bin/env python

from distutils.core import setup
import sys

if sys.version_info < (2, 7):
    print("Python 2.7 or higher required, please upgrade.")
    sys.exit(1)

setup(name="FIAT",
      version="1.4.0+",
      description="FInite element Automatic Tabulator",
      author="Robert C. Kirby",
      author_email="robert.c.kirby@gmail.com",
      url="http://www.math.ttu.edu/~kirby",
      license="LGPL v3 or later",
      packages=['FIAT'])
