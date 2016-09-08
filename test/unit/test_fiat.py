# Copyright (C) 2015-2016 Jan Blechta
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

from __future__ import absolute_import, print_function, division

import random
import numpy as np
import pytest

from FIAT.reference_element import LINE, ReferenceElement
from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.lagrange import Lagrange
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.crouzeix_raviart import CrouzeixRaviart
from FIAT.enriched import EnrichedElement
from FIAT.bubble import Bubble


def test_basis_derivatives_scaling():
    class Interval(ReferenceElement):

        def __init__(self, a, b):
            verts = ((a,), (b,))
            edges = {0: (0, 1)}
            topology = {0: {0: (0,), 1: (1,)},
                        1: edges}
            super(Interval, self).__init__(LINE, verts, topology)

    random.seed(42)
    for i in range(26):
        a = 1000.0*(random.random() - 0.5)
        b = 1000.0*(random.random() - 0.5)
        a, b = min(a, b), max(a, b)

        interval = Interval(a, b)
        element = Lagrange(interval, 1)

        points = [(a,), (0.5*(a+b),), (b,)]
        tab = element.get_nodal_basis().tabulate(points, 2)

        # first basis function
        assert np.isclose(tab[(0,)][0][0], 1.0)
        assert np.isclose(tab[(0,)][0][1], 0.5)
        assert np.isclose(tab[(0,)][0][2], 0.0)
        # second basis function
        assert np.isclose(tab[(0,)][1][0], 0.0)
        assert np.isclose(tab[(0,)][1][1], 0.5)
        assert np.isclose(tab[(0,)][1][2], 1.0)

        # first and second derivatives
        D = 1.0 / (b - a)
        for p in range(len(points)):
            assert np.isclose(tab[(1,)][0][p], -D)
            assert np.isclose(tab[(1,)][1][p], +D)
            assert np.isclose(tab[(2,)][0][p], 0.0)
            assert np.isclose(tab[(2,)][1][p], 0.0)


@pytest.mark.parametrize(('element',), [
    # Basic elements
    (Lagrange(UFCInterval(), 1),),
    (Lagrange(UFCInterval(), 2),),
    (Lagrange(UFCInterval(), 3),),
    (Lagrange(UFCTriangle(), 1),),
    (Lagrange(UFCTriangle(), 2),),
    (Lagrange(UFCTriangle(), 3),),
    (Lagrange(UFCTetrahedron(), 1),),
    (Lagrange(UFCTetrahedron(), 2),),
    (Lagrange(UFCTetrahedron(), 3),),
    (DiscontinuousLagrange(UFCInterval(), 0),),
    (DiscontinuousLagrange(UFCInterval(), 1),),
    (DiscontinuousLagrange(UFCInterval(), 2),),
    (DiscontinuousLagrange(UFCTriangle(), 0),),
    (DiscontinuousLagrange(UFCTriangle(), 1),),
    (DiscontinuousLagrange(UFCTriangle(), 2),),
    (DiscontinuousLagrange(UFCTetrahedron(), 0),),
    (DiscontinuousLagrange(UFCTetrahedron(), 1),),
    (DiscontinuousLagrange(UFCTetrahedron(), 2),),
    (CrouzeixRaviart(UFCInterval(), 1),),
    (CrouzeixRaviart(UFCTriangle(), 1),),
    (CrouzeixRaviart(UFCTetrahedron(), 1),),

    # FIXME: for non-affine mapped elements test does not work
    #(RaviartThomas(UFCTriangle(), 1),),
    #(BrezziDouglasMarini(UFCTriangle(), 1),),
    #(Nedelec(UFCTriangle(), 1),),
    #(NedelecSecondKind(UFCTriangle(), 1),),
    #(Regge(UFCTriangle(), 1),),
    #(HHJ(UFCTriangle(), 1),),

    # Compound elements
    (EnrichedElement(Lagrange(UFCTriangle(), 1), Bubble(UFCTriangle(), 3)),),
    (EnrichedElement(Lagrange(UFCTetrahedron(), 1), Bubble(UFCTetrahedron(), 4)),),
    ])
def test_nodality(element):
    poly_set = element.get_nodal_basis()
    dual_set = element.get_dual_set()
    assert poly_set.get_reference_element() == dual_set.get_reference_element()

    coeffs_poly = poly_set.get_coeffs()
    coeffs_dual = dual_set.to_riesz(poly_set)
    assert coeffs_poly.shape == coeffs_dual.shape

    for i in range(coeffs_dual.shape[0]):
        for j in range(coeffs_poly.shape[0]):
            assert np.isclose(coeffs_dual[i].dot(coeffs_poly[j]), 1.0 if i==j else 0.0)


@pytest.mark.parametrize(('elements',), [
    ((Lagrange(UFCInterval(), 2), Bubble(UFCInterval(), 2)),),
    ((Lagrange(UFCTriangle(), 3), Bubble(UFCTriangle(), 3)),),
    ((Lagrange(UFCTetrahedron(), 4), Bubble(UFCTetrahedron(), 4)),),
    ((Lagrange(UFCInterval(), 1), Lagrange(UFCInterval(), 1)),),
    ((Lagrange(UFCInterval(), 1), Bubble(UFCInterval(), 2), Bubble(UFCInterval(), 2)),),
    ])
def test_illposed_enriched(elements):
    with pytest.raises(np.linalg.LinAlgError):
        EnrichedElement(*elements)


def test_empty_bubble():
    with pytest.raises(RuntimeError):
        Bubble(UFCInterval(), 1)
    with pytest.raises(RuntimeError):
        Bubble(UFCTriangle(), 2)
    with pytest.raises(RuntimeError):
        Bubble(UFCTetrahedron(), 3)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
