# Copyright (C) 2015-2016 Imperial College London and others
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
# Andrew McRae

import pytest
import numpy as np

from FIAT.reference_element import UFCInterval, UFCTriangle
from FIAT.lagrange import Lagrange
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.nedelec import Nedelec
from FIAT.raviart_thomas import RaviartThomas
from FIAT.tensor_product import TensorProductElement, FlattenedDimensions
from FIAT.hdivcurl import Hdiv, Hcurl
from FIAT.enriched import EnrichedElement


def test_TFE_1Dx1D_scalar():
    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(T, 1)
    P2 = Lagrange(T, 2)

    elt = TensorProductElement(P1_DG, P2)
    assert elt.value_shape() == ()
    tab = elt.tabulate(1, [(0.1, 0.2)])
    tabA = P1_DG.tabulate(1, [(0.1,)])
    tabB = P2.tabulate(1, [(0.2,)])
    for da, db in [[(0,), (0,)], [(1,), (0,)], [(0,), (1,)]]:
        dc = da + db
        assert np.isclose(tab[dc][0][0], tabA[da][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][0], tabA[da][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][0], tabA[da][0][0]*tabB[db][2][0])
        assert np.isclose(tab[dc][3][0], tabA[da][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][4][0], tabA[da][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][5][0], tabA[da][1][0]*tabB[db][2][0])


def test_TFE_1Dx1D_vector():
    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(T, 1)
    P2 = Lagrange(T, 2)

    elt = TensorProductElement(P1_DG, P2)
    hdiv_elt = Hdiv(elt)
    hcurl_elt = Hcurl(elt)
    assert hdiv_elt.value_shape() == (2,)
    assert hcurl_elt.value_shape() == (2,)

    tabA = P1_DG.tabulate(1, [(0.1,)])
    tabB = P2.tabulate(1, [(0.2,)])

    hdiv_tab = hdiv_elt.tabulate(1, [(0.1, 0.2)])
    for da, db in [[(0,), (0,)], [(1,), (0,)], [(0,), (1,)]]:
        dc = da + db
        assert hdiv_tab[dc][0][0][0] == 0.0
        assert hdiv_tab[dc][1][0][0] == 0.0
        assert hdiv_tab[dc][2][0][0] == 0.0
        assert hdiv_tab[dc][3][0][0] == 0.0
        assert hdiv_tab[dc][4][0][0] == 0.0
        assert hdiv_tab[dc][5][0][0] == 0.0
        assert np.isclose(hdiv_tab[dc][0][1][0], tabA[da][0][0]*tabB[db][0][0])
        assert np.isclose(hdiv_tab[dc][1][1][0], tabA[da][0][0]*tabB[db][1][0])
        assert np.isclose(hdiv_tab[dc][2][1][0], tabA[da][0][0]*tabB[db][2][0])
        assert np.isclose(hdiv_tab[dc][3][1][0], tabA[da][1][0]*tabB[db][0][0])
        assert np.isclose(hdiv_tab[dc][4][1][0], tabA[da][1][0]*tabB[db][1][0])
        assert np.isclose(hdiv_tab[dc][5][1][0], tabA[da][1][0]*tabB[db][2][0])

    hcurl_tab = hcurl_elt.tabulate(1, [(0.1, 0.2)])
    for da, db in [[(0,), (0,)], [(1,), (0,)], [(0,), (1,)]]:
        dc = da + db
        assert np.isclose(hcurl_tab[dc][0][0][0], tabA[da][0][0]*tabB[db][0][0])
        assert np.isclose(hcurl_tab[dc][1][0][0], tabA[da][0][0]*tabB[db][1][0])
        assert np.isclose(hcurl_tab[dc][2][0][0], tabA[da][0][0]*tabB[db][2][0])
        assert np.isclose(hcurl_tab[dc][3][0][0], tabA[da][1][0]*tabB[db][0][0])
        assert np.isclose(hcurl_tab[dc][4][0][0], tabA[da][1][0]*tabB[db][1][0])
        assert np.isclose(hcurl_tab[dc][5][0][0], tabA[da][1][0]*tabB[db][2][0])
        assert hcurl_tab[dc][0][1][0] == 0.0
        assert hcurl_tab[dc][1][1][0] == 0.0
        assert hcurl_tab[dc][2][1][0] == 0.0
        assert hcurl_tab[dc][3][1][0] == 0.0
        assert hcurl_tab[dc][4][1][0] == 0.0
        assert hcurl_tab[dc][5][1][0] == 0.0


def test_TFE_2Dx1D_scalar_triangle():
    S = UFCTriangle()
    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(S, 1)
    P2 = Lagrange(T, 2)

    elt = TensorProductElement(P1_DG, P2)
    assert elt.value_shape() == ()
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = P1_DG.tabulate(1, [(0.1, 0.2)])
    tabB = P2.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert np.isclose(tab[dc][0][0], tabA[da][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][0], tabA[da][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][0], tabA[da][0][0]*tabB[db][2][0])
        assert np.isclose(tab[dc][3][0], tabA[da][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][4][0], tabA[da][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][5][0], tabA[da][1][0]*tabB[db][2][0])
        assert np.isclose(tab[dc][6][0], tabA[da][2][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][7][0], tabA[da][2][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][8][0], tabA[da][2][0]*tabB[db][2][0])


def test_TFE_2Dx1D_scalar_quad():
    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = TensorProductElement(TensorProductElement(P1, P1_DG), P1)
    assert elt.value_shape() == ()
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1.tabulate(1, [(0.1,)])
    tB = P1_DG.tabulate(1, [(0.2,)])
    tC = P1.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert np.isclose(tab[dd][0][0], tA[da][0][0]*tB[db][0][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][1][0], tA[da][0][0]*tB[db][0][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][3][0], tA[da][0][0]*tB[db][1][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][4][0], tA[da][1][0]*tB[db][0][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][5][0], tA[da][1][0]*tB[db][0][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][6][0], tA[da][1][0]*tB[db][1][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][7][0], tA[da][1][0]*tB[db][1][0]*tC[dc][1][0])


def test_TFE_2Dx1D_scalar_triangle_hdiv():
    S = UFCTriangle()
    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(S, 1)
    P2 = Lagrange(T, 2)

    elt = Hdiv(TensorProductElement(P1_DG, P2))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = P1_DG.tabulate(1, [(0.1, 0.2)])
    tabB = P2.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert tab[dc][0][0][0] == 0.0
        assert tab[dc][1][0][0] == 0.0
        assert tab[dc][2][0][0] == 0.0
        assert tab[dc][3][0][0] == 0.0
        assert tab[dc][4][0][0] == 0.0
        assert tab[dc][5][0][0] == 0.0
        assert tab[dc][6][0][0] == 0.0
        assert tab[dc][7][0][0] == 0.0
        assert tab[dc][8][0][0] == 0.0
        assert tab[dc][0][1][0] == 0.0
        assert tab[dc][1][1][0] == 0.0
        assert tab[dc][2][1][0] == 0.0
        assert tab[dc][3][1][0] == 0.0
        assert tab[dc][4][1][0] == 0.0
        assert tab[dc][5][1][0] == 0.0
        assert tab[dc][6][1][0] == 0.0
        assert tab[dc][7][1][0] == 0.0
        assert tab[dc][8][1][0] == 0.0
        assert np.isclose(tab[dc][0][2][0], tabA[da][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][2][0], tabA[da][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][2][0], tabA[da][0][0]*tabB[db][2][0])
        assert np.isclose(tab[dc][3][2][0], tabA[da][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][4][2][0], tabA[da][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][5][2][0], tabA[da][1][0]*tabB[db][2][0])
        assert np.isclose(tab[dc][6][2][0], tabA[da][2][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][7][2][0], tabA[da][2][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][8][2][0], tabA[da][2][0]*tabB[db][2][0])


def test_TFE_2Dx1D_scalar_triangle_hcurl():
    S = UFCTriangle()
    T = UFCInterval()
    P1 = Lagrange(S, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hcurl(TensorProductElement(P1, P1_DG))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = P1.tabulate(1, [(0.1, 0.2)])
    tabB = P1_DG.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert tab[dc][0][0][0] == 0.0
        assert tab[dc][1][0][0] == 0.0
        assert tab[dc][2][0][0] == 0.0
        assert tab[dc][3][0][0] == 0.0
        assert tab[dc][4][0][0] == 0.0
        assert tab[dc][5][0][0] == 0.0
        assert tab[dc][0][1][0] == 0.0
        assert tab[dc][1][1][0] == 0.0
        assert tab[dc][2][1][0] == 0.0
        assert tab[dc][3][1][0] == 0.0
        assert tab[dc][4][1][0] == 0.0
        assert tab[dc][5][1][0] == 0.0
        assert np.isclose(tab[dc][0][2][0], tabA[da][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][2][0], tabA[da][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][2][0], tabA[da][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][2][0], tabA[da][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][2][0], tabA[da][2][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][2][0], tabA[da][2][0]*tabB[db][1][0])


def test_TFE_2Dx1D_scalar_quad_hdiv():
    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hdiv(TensorProductElement(TensorProductElement(P1_DG, P1_DG), P1))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1_DG.tabulate(1, [(0.1,)])
    tB = P1_DG.tabulate(1, [(0.2,)])
    tC = P1.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert tab[dd][0][0][0] == 0.0
        assert tab[dd][1][0][0] == 0.0
        assert tab[dd][2][0][0] == 0.0
        assert tab[dd][3][0][0] == 0.0
        assert tab[dd][4][0][0] == 0.0
        assert tab[dd][5][0][0] == 0.0
        assert tab[dd][6][0][0] == 0.0
        assert tab[dd][7][0][0] == 0.0
        assert tab[dd][0][1][0] == 0.0
        assert tab[dd][1][1][0] == 0.0
        assert tab[dd][2][1][0] == 0.0
        assert tab[dd][3][1][0] == 0.0
        assert tab[dd][4][1][0] == 0.0
        assert tab[dd][5][1][0] == 0.0
        assert tab[dd][6][1][0] == 0.0
        assert tab[dd][7][1][0] == 0.0
        assert np.isclose(tab[dd][0][2][0], tA[da][0][0]*tB[db][0][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][1][2][0], tA[da][0][0]*tB[db][0][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][2][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][3][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][4][2][0], tA[da][1][0]*tB[db][0][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][5][2][0], tA[da][1][0]*tB[db][0][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][6][2][0], tA[da][1][0]*tB[db][1][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][7][2][0], tA[da][1][0]*tB[db][1][0]*tC[dc][1][0])


def test_TFE_2Dx1D_scalar_quad_hcurl():
    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hcurl(TensorProductElement(TensorProductElement(P1, P1), P1_DG))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1.tabulate(1, [(0.1,)])
    tB = P1.tabulate(1, [(0.2,)])
    tC = P1_DG.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert tab[dd][0][0][0] == 0.0
        assert tab[dd][1][0][0] == 0.0
        assert tab[dd][2][0][0] == 0.0
        assert tab[dd][3][0][0] == 0.0
        assert tab[dd][4][0][0] == 0.0
        assert tab[dd][5][0][0] == 0.0
        assert tab[dd][6][0][0] == 0.0
        assert tab[dd][7][0][0] == 0.0
        assert tab[dd][0][1][0] == 0.0
        assert tab[dd][1][1][0] == 0.0
        assert tab[dd][2][1][0] == 0.0
        assert tab[dd][3][1][0] == 0.0
        assert tab[dd][4][1][0] == 0.0
        assert tab[dd][5][1][0] == 0.0
        assert tab[dd][6][1][0] == 0.0
        assert tab[dd][7][1][0] == 0.0
        assert np.isclose(tab[dd][0][2][0], tA[da][0][0]*tB[db][0][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][1][2][0], tA[da][0][0]*tB[db][0][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][2][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][3][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][4][2][0], tA[da][1][0]*tB[db][0][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][5][2][0], tA[da][1][0]*tB[db][0][0]*tC[dc][1][0])
        assert np.isclose(tab[dd][6][2][0], tA[da][1][0]*tB[db][1][0]*tC[dc][0][0])
        assert np.isclose(tab[dd][7][2][0], tA[da][1][0]*tB[db][1][0]*tC[dc][1][0])


def test_TFE_2Dx1D_vector_triangle_hdiv():
    S = UFCTriangle()
    T = UFCInterval()
    RT1 = RaviartThomas(S, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hdiv(TensorProductElement(RT1, P1_DG))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = RT1.tabulate(1, [(0.1, 0.2)])
    tabB = P1_DG.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert np.isclose(tab[dc][0][0][0], tabA[da][0][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][0][0], tabA[da][0][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][0][0], tabA[da][1][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][0][0], tabA[da][1][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][0][0], tabA[da][2][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][0][0], tabA[da][2][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][0][1][0], tabA[da][0][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][1][0], tabA[da][0][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][1][0], tabA[da][1][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][1][0], tabA[da][1][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][1][0], tabA[da][2][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][1][0], tabA[da][2][1][0]*tabB[db][1][0])
        assert tab[dc][0][2][0] == 0.0
        assert tab[dc][1][2][0] == 0.0
        assert tab[dc][2][2][0] == 0.0
        assert tab[dc][3][2][0] == 0.0
        assert tab[dc][4][2][0] == 0.0
        assert tab[dc][5][2][0] == 0.0


def test_TFE_2Dx1D_vector_triangle_hcurl():
    S = UFCTriangle()
    T = UFCInterval()
    Ned1 = Nedelec(S, 1)
    P1 = Lagrange(T, 1)

    elt = Hcurl(TensorProductElement(Ned1, P1))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = Ned1.tabulate(1, [(0.1, 0.2)])
    tabB = P1.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert np.isclose(tab[dc][0][0][0], tabA[da][0][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][0][0], tabA[da][0][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][0][0], tabA[da][1][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][0][0], tabA[da][1][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][0][0], tabA[da][2][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][0][0], tabA[da][2][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][0][1][0], tabA[da][0][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][1][0], tabA[da][0][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][1][0], tabA[da][1][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][1][0], tabA[da][1][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][1][0], tabA[da][2][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][1][0], tabA[da][2][1][0]*tabB[db][1][0])
        assert tab[dc][0][2][0] == 0.0
        assert tab[dc][1][2][0] == 0.0
        assert tab[dc][2][2][0] == 0.0
        assert tab[dc][3][2][0] == 0.0
        assert tab[dc][4][2][0] == 0.0
        assert tab[dc][5][2][0] == 0.0


def test_TFE_2Dx1D_vector_triangle_hdiv_rotate():
    S = UFCTriangle()
    T = UFCInterval()
    Ned1 = Nedelec(S, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hdiv(TensorProductElement(Ned1, P1_DG))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = Ned1.tabulate(1, [(0.1, 0.2)])
    tabB = P1_DG.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert np.isclose(tab[dc][0][0][0], tabA[da][0][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][0][0], tabA[da][0][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][0][0], tabA[da][1][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][0][0], tabA[da][1][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][0][0], tabA[da][2][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][0][0], tabA[da][2][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][0][1][0], -tabA[da][0][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][1][0], -tabA[da][0][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][1][0], -tabA[da][1][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][1][0], -tabA[da][1][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][1][0], -tabA[da][2][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][1][0], -tabA[da][2][0][0]*tabB[db][1][0])
        assert tab[dc][0][2][0] == 0.0
        assert tab[dc][1][2][0] == 0.0
        assert tab[dc][2][2][0] == 0.0
        assert tab[dc][3][2][0] == 0.0
        assert tab[dc][4][2][0] == 0.0
        assert tab[dc][5][2][0] == 0.0


def test_TFE_2Dx1D_vector_triangle_hcurl_rotate():
    S = UFCTriangle()
    T = UFCInterval()
    RT1 = RaviartThomas(S, 1)
    P1 = Lagrange(T, 1)

    elt = Hcurl(TensorProductElement(RT1, P1))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = RT1.tabulate(1, [(0.1, 0.2)])
    tabB = P1.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert np.isclose(tab[dc][0][0][0], -tabA[da][0][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][0][0], -tabA[da][0][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][0][0], -tabA[da][1][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][0][0], -tabA[da][1][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][0][0], -tabA[da][2][1][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][0][0], -tabA[da][2][1][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][0][1][0], tabA[da][0][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][1][1][0], tabA[da][0][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][2][1][0], tabA[da][1][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][3][1][0], tabA[da][1][0][0]*tabB[db][1][0])
        assert np.isclose(tab[dc][4][1][0], tabA[da][2][0][0]*tabB[db][0][0])
        assert np.isclose(tab[dc][5][1][0], tabA[da][2][0][0]*tabB[db][1][0])
        assert tab[dc][0][2][0] == 0.0
        assert tab[dc][1][2][0] == 0.0
        assert tab[dc][2][2][0] == 0.0
        assert tab[dc][3][2][0] == 0.0
        assert tab[dc][4][2][0] == 0.0
        assert tab[dc][5][2][0] == 0.0


def test_TFE_2Dx1D_vector_quad_hdiv():
    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P0 = DiscontinuousLagrange(T, 0)
    P1_DG = DiscontinuousLagrange(T, 1)

    P1P0 = Hdiv(TensorProductElement(P1, P0))
    P0P1 = Hdiv(TensorProductElement(P0, P1))
    horiz_elt = EnrichedElement(P1P0, P0P1)
    elt = Hdiv(TensorProductElement(horiz_elt, P1_DG))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1.tabulate(1, [(0.1,)])
    tB = P0.tabulate(1, [(0.2,)])
    tC = P0.tabulate(1, [(0.1,)])
    tD = P1.tabulate(1, [(0.2,)])
    tE = P1_DG.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert np.isclose(tab[dd][0][0][0], -tA[da][0][0]*tB[db][0][0]*tE[dc][0][0])
        assert np.isclose(tab[dd][1][0][0], -tA[da][0][0]*tB[db][0][0]*tE[dc][1][0])
        assert np.isclose(tab[dd][2][0][0], -tA[da][1][0]*tB[db][0][0]*tE[dc][0][0])
        assert np.isclose(tab[dd][3][0][0], -tA[da][1][0]*tB[db][0][0]*tE[dc][1][0])
        assert tab[dd][4][0][0] == 0.0
        assert tab[dd][5][0][0] == 0.0
        assert tab[dd][6][0][0] == 0.0
        assert tab[dd][7][0][0] == 0.0
        assert tab[dd][0][1][0] == 0.0
        assert tab[dd][1][1][0] == 0.0
        assert tab[dd][2][1][0] == 0.0
        assert tab[dd][3][1][0] == 0.0
        assert np.isclose(tab[dd][4][1][0], tC[da][0][0]*tD[db][0][0]*tE[dc][0][0])
        assert np.isclose(tab[dd][5][1][0], tC[da][0][0]*tD[db][0][0]*tE[dc][1][0])
        assert np.isclose(tab[dd][6][1][0], tC[da][0][0]*tD[db][1][0]*tE[dc][0][0])
        assert np.isclose(tab[dd][7][1][0], tC[da][0][0]*tD[db][1][0]*tE[dc][1][0])
        assert tab[dd][0][2][0] == 0.0
        assert tab[dd][1][2][0] == 0.0
        assert tab[dd][2][2][0] == 0.0
        assert tab[dd][3][2][0] == 0.0
        assert tab[dd][4][2][0] == 0.0
        assert tab[dd][5][2][0] == 0.0
        assert tab[dd][6][2][0] == 0.0
        assert tab[dd][7][2][0] == 0.0


def test_TFE_2Dx1D_vector_quad_hcurl():
    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P0 = DiscontinuousLagrange(T, 0)

    P1P0 = Hcurl(TensorProductElement(P1, P0))
    P0P1 = Hcurl(TensorProductElement(P0, P1))
    horiz_elt = EnrichedElement(P1P0, P0P1)
    elt = Hcurl(TensorProductElement(horiz_elt, P1))
    assert elt.value_shape() == (3,)
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1.tabulate(1, [(0.1,)])
    tB = P0.tabulate(1, [(0.2,)])
    tC = P0.tabulate(1, [(0.1,)])
    tD = P1.tabulate(1, [(0.2,)])
    tE = P1.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert tab[dd][0][0][0] == 0.0
        assert tab[dd][1][0][0] == 0.0
        assert tab[dd][2][0][0] == 0.0
        assert tab[dd][3][0][0] == 0.0
        assert np.isclose(tab[dd][4][0][0], tC[da][0][0]*tD[db][0][0]*tE[dc][0][0])
        assert np.isclose(tab[dd][5][0][0], tC[da][0][0]*tD[db][0][0]*tE[dc][1][0])
        assert np.isclose(tab[dd][6][0][0], tC[da][0][0]*tD[db][1][0]*tE[dc][0][0])
        assert np.isclose(tab[dd][7][0][0], tC[da][0][0]*tD[db][1][0]*tE[dc][1][0])
        assert np.isclose(tab[dd][0][1][0], tA[da][0][0]*tB[db][0][0]*tE[dc][0][0])
        assert np.isclose(tab[dd][1][1][0], tA[da][0][0]*tB[db][0][0]*tE[dc][1][0])
        assert np.isclose(tab[dd][2][1][0], tA[da][1][0]*tB[db][0][0]*tE[dc][0][0])
        assert np.isclose(tab[dd][3][1][0], tA[da][1][0]*tB[db][0][0]*tE[dc][1][0])
        assert tab[dd][4][1][0] == 0.0
        assert tab[dd][5][1][0] == 0.0
        assert tab[dd][6][1][0] == 0.0
        assert tab[dd][7][1][0] == 0.0
        assert tab[dd][0][2][0] == 0.0
        assert tab[dd][1][2][0] == 0.0
        assert tab[dd][2][2][0] == 0.0
        assert tab[dd][3][2][0] == 0.0
        assert tab[dd][4][2][0] == 0.0
        assert tab[dd][5][2][0] == 0.0
        assert tab[dd][6][2][0] == 0.0
        assert tab[dd][7][2][0] == 0.0


def test_flattened_against_tpe_quad():
    T = UFCInterval()
    P1 = Lagrange(T, 1)
    tpe_quad = TensorProductElement(P1, P1)
    flattened_quad = FlattenedDimensions(tpe_quad)
    assert tpe_quad.value_shape() == ()
    tpe_tab = tpe_quad.tabulate(1, [(0.1, 0.2)])
    flattened_tab = flattened_quad.tabulate(1, [(0.1, 0.2)])

    for da, db in [[(0,), (0,)], [(1,), (0,)], [(0,), (1,)]]:
        dc = da + db
        assert np.isclose(tpe_tab[dc][0][0], flattened_tab[dc][0][0])
        assert np.isclose(tpe_tab[dc][1][0], flattened_tab[dc][1][0])
        assert np.isclose(tpe_tab[dc][2][0], flattened_tab[dc][2][0])
        assert np.isclose(tpe_tab[dc][3][0], flattened_tab[dc][3][0])


def test_flattened_against_tpe_hex():
    T = UFCInterval()
    P1 = Lagrange(T, 1)
    tpe_quad = TensorProductElement(P1, P1)
    tpe_hex = TensorProductElement(tpe_quad, P1)
    flattened_quad = FlattenedDimensions(tpe_quad)
    flattened_hex = FlattenedDimensions(TensorProductElement(flattened_quad, P1))
    assert tpe_quad.value_shape() == ()
    tpe_tab = tpe_hex.tabulate(1, [(0.1, 0.2, 0.3)])
    flattened_tab = flattened_hex.tabulate(1, [(0.1, 0.2, 0.3)])

    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert np.isclose(tpe_tab[dd][0][0], flattened_tab[dd][0][0])
        assert np.isclose(tpe_tab[dd][1][0], flattened_tab[dd][1][0])
        assert np.isclose(tpe_tab[dd][2][0], flattened_tab[dd][2][0])
        assert np.isclose(tpe_tab[dd][3][0], flattened_tab[dd][3][0])
        assert np.isclose(tpe_tab[dd][4][0], flattened_tab[dd][4][0])
        assert np.isclose(tpe_tab[dd][5][0], flattened_tab[dd][5][0])
        assert np.isclose(tpe_tab[dd][6][0], flattened_tab[dd][6][0])
        assert np.isclose(tpe_tab[dd][7][0], flattened_tab[dd][7][0])


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
