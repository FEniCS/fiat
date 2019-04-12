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
# David Ham and Cyrus Cheng

import pytest
import numpy as np
from FIAT.reference_element import *


hypercube_simplex_map = {Point(): Point(),
                         DefaultLine(): DefaultLine(),
                         UFCInterval(): UFCInterval(),
                         UFCQuadrilateral(): UFCTriangle(),
                         UFCHexahedron(): UFCTetrahedron()}


@pytest.mark.parametrize("dim, degree", [(dim, degree)
                                         for dim in range(1, 3)
                                         for degree in range(1, 6)])
def test_basis_values(dim, degree):
    """Ensure that integrating a simple monomial produces the expected results."""
    from FIAT import ufc_cell, make_quadrature
    from FIAT.serendipity import Serendipity

    cell = np.array([None, 'interval', 'quadrilateral', 'hexahedron'])
    s = ufc_cell(cell[dim])
    q = make_quadrature(s, degree + 1)

    fe = Serendipity(s, degree)
    tab = fe.tabulate(0, q.pts)[(0,) * dim]

    #points = [(0,0), (0,1), (1,0), (1,1), (0,0.5), (1,0.5), (0.5,0), (0.5,1)]

    l = np.shape(tab)[0]

    for test_degree in range(degree + 1):
        coefs = [p[0]**test_degree for p in q.pts]
        integral = np.float(np.dot(coefs[:l], np.dot(tab, q.wts)))
        reference = np.dot([x[0]**test_degree
                            for x in q.pts], q.wts)
        assert np.isclose(integral, reference, rtol=1e-14)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
