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
# David Ham

import pytest
import numpy as np


@pytest.mark.parametrize("dim, degree", [(dim, degree)
                                         for dim in range(1, 4)
                                         for degree in range(4)])
def test_basis_values(dim, degree):
    """Ensure that integrating a simple monomial produces the expected results."""
    from FIAT import ufc_simplex, DiscontinuousTaylor, make_quadrature

    s = ufc_simplex(dim)
    q = make_quadrature(s, degree + 1)

    fe = DiscontinuousTaylor(s, degree)
    tab = fe.tabulate(0, q.pts)[(0,) * dim]

    for test_degree in range(degree + 1):
        coefs = [n(lambda x: x[0]**test_degree) for n in fe.dual.nodes]
        integral = np.float(np.dot(coefs, np.dot(tab, q.wts)))
        reference = np.dot([x[0]**test_degree
                            for x in q.pts], q.wts)
        assert np.isclose(integral, reference, rtol=1e-14)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
