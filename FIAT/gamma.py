# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2016

from __future__ import absolute_import, print_function, division

import math


# A C implementation for ln_gamma function taken from Numerical
# recipes in C: The art of scientific
# computing, 2nd edition, Press, Teukolsky, Vetterling, Flannery, Cambridge
# University press, page 214
# translated into Python by Robert Kirby
# See originally Abramowitz and Stegun's Handbook of Mathematical Functions.
def ln_gamma(xx):
    cof = [76.18009172947146,
           -86.50532032941677,
           24.01409824083091,
           -1.231739572450155,
           0.1208650973866179e-2,
           -0.5395239384953e-5]
    y = xx
    x = xx
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for j in range(0, 6):
        y = y + 1
        ser += cof[j] / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def gamma(xx):
    return math.exp(ln_gamma(xx))
