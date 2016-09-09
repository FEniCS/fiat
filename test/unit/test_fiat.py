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
from FIAT.discontinuous_lagrange import DiscontinuousLagrange  # noqa: F401
from FIAT.crouzeix_raviart import CrouzeixRaviart              # noqa: F401
from FIAT.raviart_thomas import RaviartThomas                  # noqa: F401
from FIAT.brezzi_douglas_marini import BrezziDouglasMarini     # noqa: F401
from FIAT.nedelec import Nedelec                               # noqa: F401
from FIAT.nedelec_second_kind import NedelecSecondKind         # noqa: F401
from FIAT.regge import Regge                                   # noqa: F401
from FIAT.tensor_product import TensorProductElement           # noqa: F401
from FIAT.bubble import Bubble
from FIAT.enriched import EnrichedElement


I = UFCInterval()
T = UFCTriangle()
S = UFCTetrahedron()


def test_basis_derivatives_scaling():
    "Regression test for issue #9"
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


@pytest.mark.parametrize('element', [
    "Lagrange(I, 1)",
    "Lagrange(I, 2)",
    "Lagrange(I, 3)",
    "Lagrange(T, 1)",
    "Lagrange(T, 2)",
    "Lagrange(T, 3)",
    "Lagrange(S, 1)",
    "Lagrange(S, 2)",
    "Lagrange(S, 3)",
    "DiscontinuousLagrange(I, 0)",
    "DiscontinuousLagrange(I, 1)",
    "DiscontinuousLagrange(I, 2)",
    "DiscontinuousLagrange(T, 0)",
    "DiscontinuousLagrange(T, 1)",
    "DiscontinuousLagrange(T, 2)",
    "DiscontinuousLagrange(S, 0)",
    "DiscontinuousLagrange(S, 1)",
    "DiscontinuousLagrange(S, 2)",
    "CrouzeixRaviart(I, 1)",
    "CrouzeixRaviart(T, 1)",
    "CrouzeixRaviart(S, 1)",
    "RaviartThomas(T, 1)",
    "RaviartThomas(T, 2)",
    "RaviartThomas(T, 3)",
    "RaviartThomas(S, 1)",
    "RaviartThomas(S, 2)",
    "RaviartThomas(S, 3)",
    "BrezziDouglasMarini(T, 1)",
    "BrezziDouglasMarini(T, 2)",
    "BrezziDouglasMarini(T, 3)",
    "BrezziDouglasMarini(S, 1)",
    "BrezziDouglasMarini(S, 2)",
    "BrezziDouglasMarini(S, 3)",
    "Nedelec(T, 1)",
    "Nedelec(T, 2)",
    "Nedelec(T, 3)",
    "Nedelec(S, 1)",
    "Nedelec(S, 2)",
    "Nedelec(S, 3)",
    "NedelecSecondKind(T, 1)",
    "NedelecSecondKind(T, 2)",
    "NedelecSecondKind(T, 3)",
    "NedelecSecondKind(S, 1)",
    "NedelecSecondKind(S, 2)",
    "NedelecSecondKind(S, 3)",
    "Regge(T, 0)",
    "Regge(T, 1)",
    "Regge(T, 2)",
    "Regge(S, 0)",
    "Regge(S, 1)",
    "Regge(S, 2)",
    # "HellanHerrmannJohnson(T, 0),)",
    # "HellanHerrmannJohnson(T, 1),)",
    # "HellanHerrmannJohnson(T, 2),)",
    "Bubble(I, 2)",
    "Bubble(T, 3)",
    "Bubble(S, 4)",
    "EnrichedElement(Lagrange(I, 1), Bubble(I, 2))",
    "EnrichedElement(Lagrange(T, 1), Bubble(T, 3))",
    "EnrichedElement(Lagrange(S, 1), Bubble(S, 4))",
    pytest.mark.xfail(strict=True)(
        "TensorProductElement(DiscontinuousLagrange(I, 1), Lagrange(I, 2))"
    ),
])
def test_nodality(element):
    """Check that generated elements are nodal, i.e. nodes evaluated
    on basis functions give Kronecker delta
    """
    # Instantiate element lazily
    element = eval(element)

    # Fetch primal and dual basis
    poly_set = element.get_nodal_basis()
    dual_set = element.get_dual_set()
    assert poly_set.get_reference_element() == dual_set.get_reference_element()

    # Get coeffs of primal and dual bases w.r.t. expansion set
    coeffs_poly = poly_set.get_coeffs()
    coeffs_dual = dual_set.to_riesz(poly_set)
    assert coeffs_poly.shape == coeffs_dual.shape

    # Check nodality
    for i in range(coeffs_dual.shape[0]):
        for j in range(coeffs_poly.shape[0]):
            assert np.isclose(
                coeffs_dual[i].flatten().dot(coeffs_poly[j].flatten()),
                1.0 if i == j else 0.0
            )


@pytest.mark.parametrize('elements', [
    (Lagrange(I, 2), Bubble(I, 2)),
    (Lagrange(T, 3), Bubble(T, 3)),
    (Lagrange(S, 4), Bubble(S, 4)),
    (Lagrange(I, 1), Lagrange(I, 1)),
    (Lagrange(I, 1), Bubble(I, 2), Bubble(I, 2)),
])
def test_illposed_enriched(elements):
    """Check that enriched element fails on ill-posed
    (non-unisolvent) case
    """
    with pytest.raises(np.linalg.LinAlgError):
        EnrichedElement(*elements)


def test_empty_bubble():
    "Check that bubble of too low degree fails"
    with pytest.raises(RuntimeError):
        Bubble(I, 1)
    with pytest.raises(RuntimeError):
        Bubble(T, 2)
    with pytest.raises(RuntimeError):
        Bubble(S, 3)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
