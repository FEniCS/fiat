# Copyright (C) 2019 Imperial College London and others
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
# Authors:
#
# Christopher Eldred

import pytest
import numpy as np
import FIAT
from FIAT.reference_element import UFCInterval


@pytest.mark.parametrize("degree", range(2, 7))
def test_egl_basis_values(degree):
    """Ensure that integrating a simple monomial produces the expected results."""
    from FIAT import ufc_simplex, ExtendedGaussLegendre, make_quadrature

    s = ufc_simplex(1)
    q = make_quadrature(s, degree + 1)

    fe = ExtendedGaussLegendre(s, degree)
    tab = fe.tabulate(0, q.pts)[(0,)]

    for test_degree in range(degree + 1):
        coefs = [n(lambda x: x[0]**test_degree) for n in fe.dual.nodes]
        integral = np.dot(coefs, np.dot(tab, q.wts))
        reference = np.dot([x[0]**test_degree
                            for x in q.pts], q.wts)
        assert np.allclose(integral, reference, rtol=1e-14)


@pytest.mark.parametrize(("points, degree"), ((p, d)
                                              for p in range(3, 10)
                                              for d in range(2*p - 7)))
def test_extended_gauss_legendre_quadrature(points, degree):
    """Check that the quadrature rules correctly integrate all the right
    polynomial degrees."""

    q = FIAT.quadrature.ExtendedGaussLegendreQuadratureLineRule(UFCInterval(), points)
    assert np.round(q.integrate(lambda x: x[0]**degree) - 1./(degree+1), 14) == 0.


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
