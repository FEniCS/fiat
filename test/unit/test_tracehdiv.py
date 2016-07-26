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
# Thomas Gibson (t.gibson15@imperial.ac.uk)

from __future__ import absolute_import, print_function, division

import pytest
import numpy as np


@pytest.mark.parametrize("degree", range(7))
@pytest.mark.parametrize("facet_id", range(3))
def test_basis_values(degree, facet_id):
    """Ensure that integrating simple monomials produces the expected results
    for each facet entity of the reference triangle."""
    from FIAT import ufc_simplex, TraceHDiv, make_quadrature

    ref_el = ufc_simplex(2)
    quadrule = make_quadrature(ufc_simplex(1), degree + 1)
    fiat_element = TraceHDiv(ref_el, degree)

    nf = fiat_element.trace.space_dimension()
    entity = (ref_el.get_spatial_dimension() - 1, facet_id)
    tab = fiat_element.tabulate(0, quadrule.pts,
                                entity)[(0, 0)][nf*facet_id:nf*(facet_id + 1)]

    for test_degree in range(degree + 1):
        coeffs = [n(lambda x: x[0]**test_degree)
                  for n in fiat_element.trace.dual.nodes]
        integral = np.dot(coeffs, np.dot(tab, quadrule.wts))
        reference = np.dot([x[0]**test_degree
                            for x in quadrule.pts], quadrule.wts)
        assert np.allclose(integral, reference, rtol=1e-14)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
