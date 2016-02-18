# Copyright (C) 2015 Jan Blechta
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

import nose
from nose.tools import assert_equal, assert_almost_equal
import numpy


def test_basis_derivatives_scaling():
    import random
    from FIAT.reference_element import LINE, ReferenceElement
    from FIAT.lagrange import Lagrange

    class Interval(ReferenceElement):
        def __init__(self, a, b):
            verts = ( (a,), (b,) )
            edges = { 0 : ( 0, 1 ) }
            topology = { 0 : { 0 : (0,) , 1: (1,) } , \
                         1 : edges }
            ReferenceElement.__init__( self, LINE, verts, topology )

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
        assert_almost_equal(tab[(0,)][0][0], 1.0)
        assert_almost_equal(tab[(0,)][0][1], 0.5)
        assert_almost_equal(tab[(0,)][0][2], 0.0)
        # second basis function
        assert_almost_equal(tab[(0,)][1][0], 0.0)
        assert_almost_equal(tab[(0,)][1][1], 0.5)
        assert_almost_equal(tab[(0,)][1][2], 1.0)

        # first and second derivatives
        D = 1.0 / (b - a)
        for p in range(len(points)):
            assert_almost_equal(tab[(1,)][0][p], -D)
            assert_almost_equal(tab[(1,)][1][p], +D)
            assert_almost_equal(tab[(2,)][0][p], 0.0)
            assert_almost_equal(tab[(2,)][1][p], 0.0)


def test_TFE_1Dx1D_scalar():
    from FIAT.reference_element import UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement

    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(T, 1)
    P2 = Lagrange(T, 2)

    elt = TensorFiniteElement(P1_DG, P2)
    assert_equal(elt.value_shape(), ())
    tab = elt.tabulate(1, [(0.1, 0.2)])
    tabA = P1_DG.tabulate(1, [(0.1,)])
    tabB = P2.tabulate(1, [(0.2,)])
    for da, db in [[(0,), (0,)], [(1,), (0,)], [(0,), (1,)]]:
        dc = da + db
        assert_almost_equal(tab[dc][0][0], tabA[da][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][0], tabA[da][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][0], tabA[da][0][0]*tabB[db][2][0])
        assert_almost_equal(tab[dc][3][0], tabA[da][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][4][0], tabA[da][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][5][0], tabA[da][1][0]*tabB[db][2][0])


def test_TFE_1Dx1D_vector():
    from FIAT.reference_element import UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hdiv, Hcurl

    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(T, 1)
    P2 = Lagrange(T, 2)

    elt = TensorFiniteElement(P1_DG, P2)
    hdiv_elt = Hdiv(elt)
    hcurl_elt = Hcurl(elt)
    assert_equal(hdiv_elt.value_shape(), (2,))
    assert_equal(hcurl_elt.value_shape(), (2,))

    tabA = P1_DG.tabulate(1, [(0.1,)])
    tabB = P2.tabulate(1, [(0.2,)])

    hdiv_tab = hdiv_elt.tabulate(1, [(0.1, 0.2)])
    for da, db in [[(0,), (0,)], [(1,), (0,)], [(0,), (1,)]]:
        dc = da + db
        assert_equal(hdiv_tab[dc][0][0][0], 0.0)
        assert_equal(hdiv_tab[dc][1][0][0], 0.0)
        assert_equal(hdiv_tab[dc][2][0][0], 0.0)
        assert_equal(hdiv_tab[dc][3][0][0], 0.0)
        assert_equal(hdiv_tab[dc][4][0][0], 0.0)
        assert_equal(hdiv_tab[dc][5][0][0], 0.0)
        assert_almost_equal(hdiv_tab[dc][0][1][0], tabA[da][0][0]*tabB[db][0][0])
        assert_almost_equal(hdiv_tab[dc][1][1][0], tabA[da][0][0]*tabB[db][1][0])
        assert_almost_equal(hdiv_tab[dc][2][1][0], tabA[da][0][0]*tabB[db][2][0])
        assert_almost_equal(hdiv_tab[dc][3][1][0], tabA[da][1][0]*tabB[db][0][0])
        assert_almost_equal(hdiv_tab[dc][4][1][0], tabA[da][1][0]*tabB[db][1][0])
        assert_almost_equal(hdiv_tab[dc][5][1][0], tabA[da][1][0]*tabB[db][2][0])

    hcurl_tab = hcurl_elt.tabulate(1, [(0.1, 0.2)])
    for da, db in [[(0,), (0,)], [(1,), (0,)], [(0,), (1,)]]:
        dc = da + db
        assert_almost_equal(hcurl_tab[dc][0][0][0], tabA[da][0][0]*tabB[db][0][0])
        assert_almost_equal(hcurl_tab[dc][1][0][0], tabA[da][0][0]*tabB[db][1][0])
        assert_almost_equal(hcurl_tab[dc][2][0][0], tabA[da][0][0]*tabB[db][2][0])
        assert_almost_equal(hcurl_tab[dc][3][0][0], tabA[da][1][0]*tabB[db][0][0])
        assert_almost_equal(hcurl_tab[dc][4][0][0], tabA[da][1][0]*tabB[db][1][0])
        assert_almost_equal(hcurl_tab[dc][5][0][0], tabA[da][1][0]*tabB[db][2][0])
        assert_equal(hcurl_tab[dc][0][1][0], 0.0)
        assert_equal(hcurl_tab[dc][1][1][0], 0.0)
        assert_equal(hcurl_tab[dc][2][1][0], 0.0)
        assert_equal(hcurl_tab[dc][3][1][0], 0.0)
        assert_equal(hcurl_tab[dc][4][1][0], 0.0)
        assert_equal(hcurl_tab[dc][5][1][0], 0.0)


def test_TFE_2Dx1D_scalar_triangle():
    from FIAT.reference_element import UFCTriangle, UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement

    S = UFCTriangle()
    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(S, 1)
    P2 = Lagrange(T, 2)

    elt = TensorFiniteElement(P1_DG, P2)
    assert_equal(elt.value_shape(), ())
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = P1_DG.tabulate(1, [(0.1, 0.2)])
    tabB = P2.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert_almost_equal(tab[dc][0][0], tabA[da][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][0], tabA[da][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][0], tabA[da][0][0]*tabB[db][2][0])
        assert_almost_equal(tab[dc][3][0], tabA[da][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][4][0], tabA[da][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][5][0], tabA[da][1][0]*tabB[db][2][0])
        assert_almost_equal(tab[dc][6][0], tabA[da][2][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][7][0], tabA[da][2][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][8][0], tabA[da][2][0]*tabB[db][2][0])


def test_TFE_2Dx1D_scalar_quad():
    from FIAT.reference_element import UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement

    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = TensorFiniteElement(TensorFiniteElement(P1, P1_DG), P1)
    assert_equal(elt.value_shape(), ())
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1.tabulate(1, [(0.1,)])
    tB = P1_DG.tabulate(1, [(0.2,)])
    tC = P1.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert_almost_equal(tab[dd][0][0], tA[da][0][0]*tB[db][0][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][1][0], tA[da][0][0]*tB[db][0][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][3][0], tA[da][0][0]*tB[db][1][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][4][0], tA[da][1][0]*tB[db][0][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][5][0], tA[da][1][0]*tB[db][0][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][6][0], tA[da][1][0]*tB[db][1][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][7][0], tA[da][1][0]*tB[db][1][0]*tC[dc][1][0])


def test_TFE_2Dx1D_scalar_triangle_hdiv():
    from FIAT.reference_element import UFCTriangle, UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hdiv

    S = UFCTriangle()
    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(S, 1)
    P2 = Lagrange(T, 2)

    elt = Hdiv(TensorFiniteElement(P1_DG, P2))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = P1_DG.tabulate(1, [(0.1, 0.2)])
    tabB = P2.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert_equal(tab[dc][0][0][0], 0.0)
        assert_equal(tab[dc][1][0][0], 0.0)
        assert_equal(tab[dc][2][0][0], 0.0)
        assert_equal(tab[dc][3][0][0], 0.0)
        assert_equal(tab[dc][4][0][0], 0.0)
        assert_equal(tab[dc][5][0][0], 0.0)
        assert_equal(tab[dc][6][0][0], 0.0)
        assert_equal(tab[dc][7][0][0], 0.0)
        assert_equal(tab[dc][8][0][0], 0.0)
        assert_equal(tab[dc][0][1][0], 0.0)
        assert_equal(tab[dc][1][1][0], 0.0)
        assert_equal(tab[dc][2][1][0], 0.0)
        assert_equal(tab[dc][3][1][0], 0.0)
        assert_equal(tab[dc][4][1][0], 0.0)
        assert_equal(tab[dc][5][1][0], 0.0)
        assert_equal(tab[dc][6][1][0], 0.0)
        assert_equal(tab[dc][7][1][0], 0.0)
        assert_equal(tab[dc][8][1][0], 0.0)
        assert_almost_equal(tab[dc][0][2][0], tabA[da][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][2][0], tabA[da][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][2][0], tabA[da][0][0]*tabB[db][2][0])
        assert_almost_equal(tab[dc][3][2][0], tabA[da][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][4][2][0], tabA[da][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][5][2][0], tabA[da][1][0]*tabB[db][2][0])
        assert_almost_equal(tab[dc][6][2][0], tabA[da][2][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][7][2][0], tabA[da][2][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][8][2][0], tabA[da][2][0]*tabB[db][2][0])


def test_TFE_2Dx1D_scalar_triangle_hcurl():
    from FIAT.reference_element import UFCTriangle, UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hcurl

    S = UFCTriangle()
    T = UFCInterval()
    P1 = Lagrange(S, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hcurl(TensorFiniteElement(P1, P1_DG))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = P1.tabulate(1, [(0.1, 0.2)])
    tabB = P1_DG.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert_equal(tab[dc][0][0][0], 0.0)
        assert_equal(tab[dc][1][0][0], 0.0)
        assert_equal(tab[dc][2][0][0], 0.0)
        assert_equal(tab[dc][3][0][0], 0.0)
        assert_equal(tab[dc][4][0][0], 0.0)
        assert_equal(tab[dc][5][0][0], 0.0)
        assert_equal(tab[dc][0][1][0], 0.0)
        assert_equal(tab[dc][1][1][0], 0.0)
        assert_equal(tab[dc][2][1][0], 0.0)
        assert_equal(tab[dc][3][1][0], 0.0)
        assert_equal(tab[dc][4][1][0], 0.0)
        assert_equal(tab[dc][5][1][0], 0.0)
        assert_almost_equal(tab[dc][0][2][0], tabA[da][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][2][0], tabA[da][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][2][0], tabA[da][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][2][0], tabA[da][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][2][0], tabA[da][2][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][2][0], tabA[da][2][0]*tabB[db][1][0])


def test_TFE_2Dx1D_scalar_quad_hdiv():
    from FIAT.reference_element import UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hdiv

    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hdiv(TensorFiniteElement(TensorFiniteElement(P1_DG, P1_DG), P1))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1_DG.tabulate(1, [(0.1,)])
    tB = P1_DG.tabulate(1, [(0.2,)])
    tC = P1.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert_equal(tab[dd][0][0][0], 0.0)
        assert_equal(tab[dd][1][0][0], 0.0)
        assert_equal(tab[dd][2][0][0], 0.0)
        assert_equal(tab[dd][3][0][0], 0.0)
        assert_equal(tab[dd][4][0][0], 0.0)
        assert_equal(tab[dd][5][0][0], 0.0)
        assert_equal(tab[dd][6][0][0], 0.0)
        assert_equal(tab[dd][7][0][0], 0.0)
        assert_equal(tab[dd][0][1][0], 0.0)
        assert_equal(tab[dd][1][1][0], 0.0)
        assert_equal(tab[dd][2][1][0], 0.0)
        assert_equal(tab[dd][3][1][0], 0.0)
        assert_equal(tab[dd][4][1][0], 0.0)
        assert_equal(tab[dd][5][1][0], 0.0)
        assert_equal(tab[dd][6][1][0], 0.0)
        assert_equal(tab[dd][7][1][0], 0.0)
        assert_almost_equal(tab[dd][0][2][0], tA[da][0][0]*tB[db][0][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][1][2][0], tA[da][0][0]*tB[db][0][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][2][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][3][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][4][2][0], tA[da][1][0]*tB[db][0][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][5][2][0], tA[da][1][0]*tB[db][0][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][6][2][0], tA[da][1][0]*tB[db][1][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][7][2][0], tA[da][1][0]*tB[db][1][0]*tC[dc][1][0])


def test_TFE_2Dx1D_scalar_quad_hcurl():
    from FIAT.reference_element import UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hcurl

    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hcurl(TensorFiniteElement(TensorFiniteElement(P1, P1), P1_DG))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1.tabulate(1, [(0.1,)])
    tB = P1.tabulate(1, [(0.2,)])
    tC = P1_DG.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert_equal(tab[dd][0][0][0], 0.0)
        assert_equal(tab[dd][1][0][0], 0.0)
        assert_equal(tab[dd][2][0][0], 0.0)
        assert_equal(tab[dd][3][0][0], 0.0)
        assert_equal(tab[dd][4][0][0], 0.0)
        assert_equal(tab[dd][5][0][0], 0.0)
        assert_equal(tab[dd][6][0][0], 0.0)
        assert_equal(tab[dd][7][0][0], 0.0)
        assert_equal(tab[dd][0][1][0], 0.0)
        assert_equal(tab[dd][1][1][0], 0.0)
        assert_equal(tab[dd][2][1][0], 0.0)
        assert_equal(tab[dd][3][1][0], 0.0)
        assert_equal(tab[dd][4][1][0], 0.0)
        assert_equal(tab[dd][5][1][0], 0.0)
        assert_equal(tab[dd][6][1][0], 0.0)
        assert_equal(tab[dd][7][1][0], 0.0)
        assert_almost_equal(tab[dd][0][2][0], tA[da][0][0]*tB[db][0][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][1][2][0], tA[da][0][0]*tB[db][0][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][2][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][3][2][0], tA[da][0][0]*tB[db][1][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][4][2][0], tA[da][1][0]*tB[db][0][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][5][2][0], tA[da][1][0]*tB[db][0][0]*tC[dc][1][0])
        assert_almost_equal(tab[dd][6][2][0], tA[da][1][0]*tB[db][1][0]*tC[dc][0][0])
        assert_almost_equal(tab[dd][7][2][0], tA[da][1][0]*tB[db][1][0]*tC[dc][1][0])


def test_TFE_2Dx1D_vector_triangle_hdiv():
    from FIAT.reference_element import UFCTriangle, UFCInterval
    from FIAT.raviart_thomas import RaviartThomas
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hdiv

    S = UFCTriangle()
    T = UFCInterval()
    RT1 = RaviartThomas(S, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hdiv(TensorFiniteElement(RT1, P1_DG))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = RT1.tabulate(1, [(0.1, 0.2)])
    tabB = P1_DG.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert_almost_equal(tab[dc][0][0][0], tabA[da][0][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][0][0], tabA[da][0][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][0][0], tabA[da][1][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][0][0], tabA[da][1][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][0][0], tabA[da][2][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][0][0], tabA[da][2][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][0][1][0], tabA[da][0][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][1][0], tabA[da][0][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][1][0], tabA[da][1][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][1][0], tabA[da][1][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][1][0], tabA[da][2][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][1][0], tabA[da][2][1][0]*tabB[db][1][0])
        assert_equal(tab[dc][0][2][0], 0.0)
        assert_equal(tab[dc][1][2][0], 0.0)
        assert_equal(tab[dc][2][2][0], 0.0)
        assert_equal(tab[dc][3][2][0], 0.0)
        assert_equal(tab[dc][4][2][0], 0.0)
        assert_equal(tab[dc][5][2][0], 0.0)


def test_TFE_2Dx1D_vector_triangle_hcurl():
    from FIAT.reference_element import UFCTriangle, UFCInterval
    from FIAT.nedelec import Nedelec
    from FIAT.lagrange import Lagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hcurl

    S = UFCTriangle()
    T = UFCInterval()
    Ned1 = Nedelec(S, 1)
    P1 = Lagrange(T, 1)

    elt = Hcurl(TensorFiniteElement(Ned1, P1))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = Ned1.tabulate(1, [(0.1, 0.2)])
    tabB = P1.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert_almost_equal(tab[dc][0][0][0], tabA[da][0][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][0][0], tabA[da][0][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][0][0], tabA[da][1][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][0][0], tabA[da][1][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][0][0], tabA[da][2][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][0][0], tabA[da][2][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][0][1][0], tabA[da][0][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][1][0], tabA[da][0][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][1][0], tabA[da][1][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][1][0], tabA[da][1][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][1][0], tabA[da][2][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][1][0], tabA[da][2][1][0]*tabB[db][1][0])
        assert_equal(tab[dc][0][2][0], 0.0)
        assert_equal(tab[dc][1][2][0], 0.0)
        assert_equal(tab[dc][2][2][0], 0.0)
        assert_equal(tab[dc][3][2][0], 0.0)
        assert_equal(tab[dc][4][2][0], 0.0)
        assert_equal(tab[dc][5][2][0], 0.0)


def test_TFE_2Dx1D_vector_triangle_hdiv_rotate():
    from FIAT.reference_element import UFCTriangle, UFCInterval
    from FIAT.nedelec import Nedelec
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hdiv

    S = UFCTriangle()
    T = UFCInterval()
    Ned1 = Nedelec(S, 1)
    P1_DG = DiscontinuousLagrange(T, 1)

    elt = Hdiv(TensorFiniteElement(Ned1, P1_DG))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = Ned1.tabulate(1, [(0.1, 0.2)])
    tabB = P1_DG.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert_almost_equal(tab[dc][0][0][0], tabA[da][0][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][0][0], tabA[da][0][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][0][0], tabA[da][1][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][0][0], tabA[da][1][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][0][0], tabA[da][2][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][0][0], tabA[da][2][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][0][1][0], -tabA[da][0][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][1][0], -tabA[da][0][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][1][0], -tabA[da][1][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][1][0], -tabA[da][1][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][1][0], -tabA[da][2][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][1][0], -tabA[da][2][0][0]*tabB[db][1][0])
        assert_equal(tab[dc][0][2][0], 0.0)
        assert_equal(tab[dc][1][2][0], 0.0)
        assert_equal(tab[dc][2][2][0], 0.0)
        assert_equal(tab[dc][3][2][0], 0.0)
        assert_equal(tab[dc][4][2][0], 0.0)
        assert_equal(tab[dc][5][2][0], 0.0)


def test_TFE_2Dx1D_vector_triangle_hcurl_rotate():
    from FIAT.reference_element import UFCTriangle, UFCInterval
    from FIAT.raviart_thomas import RaviartThomas
    from FIAT.lagrange import Lagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.hdivcurl import Hcurl

    S = UFCTriangle()
    T = UFCInterval()
    RT1 = RaviartThomas(S, 1)
    P1 = Lagrange(T, 1)

    elt = Hcurl(TensorFiniteElement(RT1, P1))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tabA = RT1.tabulate(1, [(0.1, 0.2)])
    tabB = P1.tabulate(1, [(0.3,)])
    for da, db in [[(0, 0), (0,)], [(1, 0), (0,)], [(0, 1), (0,)], [(0, 0), (1,)]]:
        dc = da + db
        assert_almost_equal(tab[dc][0][0][0], -tabA[da][0][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][0][0], -tabA[da][0][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][0][0], -tabA[da][1][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][0][0], -tabA[da][1][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][0][0], -tabA[da][2][1][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][0][0], -tabA[da][2][1][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][0][1][0], tabA[da][0][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][1][1][0], tabA[da][0][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][2][1][0], tabA[da][1][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][3][1][0], tabA[da][1][0][0]*tabB[db][1][0])
        assert_almost_equal(tab[dc][4][1][0], tabA[da][2][0][0]*tabB[db][0][0])
        assert_almost_equal(tab[dc][5][1][0], tabA[da][2][0][0]*tabB[db][1][0])
        assert_equal(tab[dc][0][2][0], 0.0)
        assert_equal(tab[dc][1][2][0], 0.0)
        assert_equal(tab[dc][2][2][0], 0.0)
        assert_equal(tab[dc][3][2][0], 0.0)
        assert_equal(tab[dc][4][2][0], 0.0)
        assert_equal(tab[dc][5][2][0], 0.0)


def test_TFE_2Dx1D_vector_quad_hdiv():
    from FIAT.reference_element import UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.enriched import EnrichedElement
    from FIAT.hdivcurl import Hdiv

    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P0 = DiscontinuousLagrange(T, 0)
    P1_DG = DiscontinuousLagrange(T, 1)

    P1P0 = Hdiv(TensorFiniteElement(P1, P0))
    P0P1 = Hdiv(TensorFiniteElement(P0, P1))
    horiz_elt = EnrichedElement(P1P0, P0P1)
    elt = Hdiv(TensorFiniteElement(horiz_elt, P1_DG))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1.tabulate(1, [(0.1,)])
    tB = P0.tabulate(1, [(0.2,)])
    tC = P0.tabulate(1, [(0.1,)])
    tD = P1.tabulate(1, [(0.2,)])
    tE = P1_DG.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert_almost_equal(tab[dd][0][0][0], -tA[da][0][0]*tB[db][0][0]*tE[dc][0][0])
        assert_almost_equal(tab[dd][1][0][0], -tA[da][0][0]*tB[db][0][0]*tE[dc][1][0])
        assert_almost_equal(tab[dd][2][0][0], -tA[da][1][0]*tB[db][0][0]*tE[dc][0][0])
        assert_almost_equal(tab[dd][3][0][0], -tA[da][1][0]*tB[db][0][0]*tE[dc][1][0])
        assert_equal(tab[dd][4][0][0], 0.0)
        assert_equal(tab[dd][5][0][0], 0.0)
        assert_equal(tab[dd][6][0][0], 0.0)
        assert_equal(tab[dd][7][0][0], 0.0)
        assert_equal(tab[dd][0][1][0], 0.0)
        assert_equal(tab[dd][1][1][0], 0.0)
        assert_equal(tab[dd][2][1][0], 0.0)
        assert_equal(tab[dd][3][1][0], 0.0)
        assert_almost_equal(tab[dd][4][1][0], tC[da][0][0]*tD[db][0][0]*tE[dc][0][0])
        assert_almost_equal(tab[dd][5][1][0], tC[da][0][0]*tD[db][0][0]*tE[dc][1][0])
        assert_almost_equal(tab[dd][6][1][0], tC[da][0][0]*tD[db][1][0]*tE[dc][0][0])
        assert_almost_equal(tab[dd][7][1][0], tC[da][0][0]*tD[db][1][0]*tE[dc][1][0])
        assert_equal(tab[dd][0][2][0], 0.0)
        assert_equal(tab[dd][1][2][0], 0.0)
        assert_equal(tab[dd][2][2][0], 0.0)
        assert_equal(tab[dd][3][2][0], 0.0)
        assert_equal(tab[dd][4][2][0], 0.0)
        assert_equal(tab[dd][5][2][0], 0.0)
        assert_equal(tab[dd][6][2][0], 0.0)
        assert_equal(tab[dd][7][2][0], 0.0)


def test_TFE_2Dx1D_vector_quad_hcurl():
    from FIAT.reference_element import UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement
    from FIAT.enriched import EnrichedElement
    from FIAT.hdivcurl import Hcurl

    T = UFCInterval()
    P1 = Lagrange(T, 1)
    P0 = DiscontinuousLagrange(T, 0)

    P1P0 = Hcurl(TensorFiniteElement(P1, P0))
    P0P1 = Hcurl(TensorFiniteElement(P0, P1))
    horiz_elt = EnrichedElement(P1P0, P0P1)
    elt = Hcurl(TensorFiniteElement(horiz_elt, P1))
    assert_equal(elt.value_shape(), (3,))
    tab = elt.tabulate(1, [(0.1, 0.2, 0.3)])
    tA = P1.tabulate(1, [(0.1,)])
    tB = P0.tabulate(1, [(0.2,)])
    tC = P0.tabulate(1, [(0.1,)])
    tD = P1.tabulate(1, [(0.2,)])
    tE = P1.tabulate(1, [(0.3,)])
    for da, db, dc in [[(0,), (0,), (0,)], [(1,), (0,), (0,)], [(0,), (1,), (0,)], [(0,), (0,), (1,)]]:
        dd = da + db + dc
        assert_equal(tab[dd][0][0][0], 0.0)
        assert_equal(tab[dd][1][0][0], 0.0)
        assert_equal(tab[dd][2][0][0], 0.0)
        assert_equal(tab[dd][3][0][0], 0.0)
        assert_almost_equal(tab[dd][4][0][0], tC[da][0][0]*tD[db][0][0]*tE[dc][0][0])
        assert_almost_equal(tab[dd][5][0][0], tC[da][0][0]*tD[db][0][0]*tE[dc][1][0])
        assert_almost_equal(tab[dd][6][0][0], tC[da][0][0]*tD[db][1][0]*tE[dc][0][0])
        assert_almost_equal(tab[dd][7][0][0], tC[da][0][0]*tD[db][1][0]*tE[dc][1][0])
        assert_almost_equal(tab[dd][0][1][0], tA[da][0][0]*tB[db][0][0]*tE[dc][0][0])
        assert_almost_equal(tab[dd][1][1][0], tA[da][0][0]*tB[db][0][0]*tE[dc][1][0])
        assert_almost_equal(tab[dd][2][1][0], tA[da][1][0]*tB[db][0][0]*tE[dc][0][0])
        assert_almost_equal(tab[dd][3][1][0], tA[da][1][0]*tB[db][0][0]*tE[dc][1][0])
        assert_equal(tab[dd][4][1][0], 0.0)
        assert_equal(tab[dd][5][1][0], 0.0)
        assert_equal(tab[dd][6][1][0], 0.0)
        assert_equal(tab[dd][7][1][0], 0.0)
        assert_equal(tab[dd][0][2][0], 0.0)
        assert_equal(tab[dd][1][2][0], 0.0)
        assert_equal(tab[dd][2][2][0], 0.0)
        assert_equal(tab[dd][3][2][0], 0.0)
        assert_equal(tab[dd][4][2][0], 0.0)
        assert_equal(tab[dd][5][2][0], 0.0)
        assert_equal(tab[dd][6][2][0], 0.0)
        assert_equal(tab[dd][7][2][0], 0.0)


if __name__ == "__main__":
    nose.main()
