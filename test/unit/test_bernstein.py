# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with FIAT.  If not, see <https://www.gnu.org/licenses/>.

import numpy
import pytest

from FIAT.reference_element import ufc_simplex
from FIAT.bernstein import Bernstein
from FIAT.quadrature_schemes import create_quadrature


D02 = numpy.array([
    [0.65423405, 1.39160021, 0.65423405, 3.95416573, 1.39160021, 3.95416573],
    [3.95416573, 3.95416573, 1.39160021, 1.39160021, 0.65423405, 0.65423405],
    [0.0831321, -2.12896637, 2.64569763, -7.25409741, 1.17096531, -6.51673126],
    [0., 0., 0., 0., 0., 0.],
    [-7.90833147, -7.90833147, -2.78320042, -2.78320042, -1.30846811, -1.30846811],
    [-2.12896637, 0.0831321, -7.25409741, 2.64569763, -6.51673126, 1.17096531],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [3.95416573, 3.95416573, 1.39160021, 1.39160021, 0.65423405, 0.65423405],
    [1.39160021, 0.65423405, 3.95416573, 0.65423405, 3.95416573, 1.39160021],
])

D11 = numpy.array([
    [0.65423405, 1.39160021, 0.65423405, 3.95416573, 1.39160021, 3.95416573],
    [3.29993168, 2.56256552, 0.73736616, -2.56256552, -0.73736616, -3.29993168],
    [0.73736616, -0.73736616, 3.29993168, -3.29993168, 2.56256552, -2.56256552],
    [-3.95416573, -3.95416573, -1.39160021, -1.39160021, -0.65423405, -0.65423405],
    [-4.69153189, -3.21679958, -4.69153189, 1.90833147, -3.21679958, 1.90833147],
    [-1.39160021, -0.65423405, -3.95416573, -0.65423405, -3.95416573, -1.39160021],
    [0., 0., 0., 0., 0., 0.],
    [3.95416573, 3.95416573, 1.39160021, 1.39160021, 0.65423405, 0.65423405],
    [1.39160021, 0.65423405, 3.95416573, 0.65423405, 3.95416573, 1.39160021],
    [0., 0., 0., 0., 0., 0.],
])

D20 = numpy.array([
    [0.65423405, 1.39160021, 0.65423405, 3.95416573, 1.39160021, 3.95416573],
    [2.64569763, 1.17096531, 0.0831321, -6.51673126, -2.12896637, -7.25409741],
    [1.39160021, 0.65423405, 3.95416573, 0.65423405, 3.95416573, 1.39160021],
    [-7.25409741, -6.51673126, -2.12896637, 1.17096531, 0.0831321, 2.64569763],
    [-2.78320042, -1.30846811, -7.90833147, -1.30846811, -7.90833147, -2.78320042],
    [0., 0., 0., 0., 0., 0.],
    [3.95416573, 3.95416573, 1.39160021, 1.39160021, 0.65423405, 0.65423405],
    [1.39160021, 0.65423405, 3.95416573, 0.65423405, 3.95416573, 1.39160021],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
])


def test_bernstein_2nd_derivatives():
    ref_el = ufc_simplex(2)
    degree = 3

    elem = Bernstein(ref_el, degree)
    rule = create_quadrature(ref_el, degree)
    points = rule.get_points()

    actual = elem.tabulate(2, points)

    assert numpy.allclose(D02, actual[(0, 2)])
    assert numpy.allclose(D11, actual[(1, 1)])
    assert numpy.allclose(D20, actual[(2, 0)])


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
