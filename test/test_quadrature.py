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

import pytest
import FIAT
import numpy


@pytest.fixture
def cell():
    return FIAT.reference_element.UFCInterval()


@pytest.mark.parametrize(("points"), range(2, 10))
def test_gauss_lobatto_quadrature(cell, points):

    q = FIAT.quadrature.GaussLobattoQuadratureLineRule(cell, points)

    for i in range(2*points - 2):
        assert numpy.round(q.integrate(lambda x: x[0]**i) - 1./(i+1), 14) == 0.

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
