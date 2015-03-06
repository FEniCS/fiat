#!/usr/bin/env python

from distutils.core import setup
import re
import sys

if sys.version_info < (2, 7):
    print("Python 2.7 or higher required, please upgrade.")
    sys.exit(1)

version = re.findall('__version__ = "(.*)"',
                     open('FIAT/__init__.py', 'r').read())[0]

setup(name="FIAT",
      version=version,
      description="FInite element Automatic Tabulator",
      author="Robert C. Kirby",
      author_email="robert.c.kirby@gmail.com",
      url="http://www.math.ttu.edu/~kirby",
      license="LGPL v3 or later",
      packages=['FIAT'])
