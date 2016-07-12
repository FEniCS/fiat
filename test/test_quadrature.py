# Copyright (C) 2015 Imperial College London and others.
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
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015

from __future__ import absolute_import

import pytest
import FIAT
import numpy


@pytest.fixture
def interval():
    return FIAT.reference_element.UFCInterval()


@pytest.mark.parametrize(("points, degree"), ((p, d)
                                              for p in range(2, 10)
                                              for d in range(2*p - 2)))
def test_gauss_lobatto_quadrature(interval, points, degree):
    """Check that the quadrature rules correctly integrate all the right
    polynomial degrees."""

    q = FIAT.quadrature.GaussLobattoQuadratureLineRule(interval, points)

    assert numpy.round(q.integrate(lambda x: x[0]**degree) - 1./(degree+1), 14) == 0.


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
