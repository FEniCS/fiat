# Copyright (C) 2016 Imperial College London and others
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
import math
import sympy
import FIAT

x = sympy.symbols('x')


@pytest.mark.parametrize("degree", list(range(0, 7)))
def test_gll_edge_basis_values(degree):
    """Ensure that edge elements are histopolatory"""

    s = FIAT.ufc_simplex(1)

    fe = FIAT.EdgeGaussLobattoLegendre(s, degree)

    basis = fe.basis[(0,)]
    cont_pts = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule(s, degree + 2).pts

    for i in range(len(cont_pts)-1):
        for j in range(basis.shape[0]):
            int_sub = sympy.integrate(basis[j], (x, cont_pts[i][0], cont_pts[i+1][0]))
            if i == j:
                assert(math.isclose(int_sub, 1.))
            else:
                assert(math.isclose(int_sub, 0., abs_tol=1e-9))


@pytest.mark.parametrize("degree", list(range(2, 7)))
def test_egl_edge_basis_values(degree):
    """Ensure that edge elements are histopolatory"""

    s = FIAT.ufc_simplex(1)

    fe = FIAT.EdgeExtendedGaussLegendre(s, degree)

    basis = fe.basis[(0,)]
    cont_pts = FIAT.quadrature.ExtendedGaussLegendreQuadratureLineRule(s, degree + 2).pts

    for i in range(len(cont_pts)-1):
        for j in range(basis.shape[0]):
            int_sub = sympy.integrate(basis[j], (x, cont_pts[i][0], cont_pts[i+1][0]))
            if i == j:
                assert(math.isclose(int_sub, 1.))
            else:
                assert(math.isclose(int_sub, 0., abs_tol=1e-9))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
