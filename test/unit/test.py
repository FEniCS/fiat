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
        nose.tools.assert_almost_equal(tab[(0,)][0][0], 1.0)
        nose.tools.assert_almost_equal(tab[(0,)][0][1], 0.5)
        nose.tools.assert_almost_equal(tab[(0,)][0][2], 0.0)
        # second basis function
        nose.tools.assert_almost_equal(tab[(0,)][1][0], 0.0)
        nose.tools.assert_almost_equal(tab[(0,)][1][1], 0.5)
        nose.tools.assert_almost_equal(tab[(0,)][1][2], 1.0)

        # first and second derivatives
        D = 1.0 / (b - a)
        for p in range(len(points)):
            nose.tools.assert_almost_equal(tab[(1,)][0][p], -D)
            nose.tools.assert_almost_equal(tab[(1,)][1][p], +D)
            nose.tools.assert_almost_equal(tab[(2,)][0][p], 0.0)
            nose.tools.assert_almost_equal(tab[(2,)][1][p], 0.0)


def test_TFE_1Dx1D_scalar():
    from FIAT.reference_element import UFCInterval
    from FIAT.lagrange import Lagrange
    from FIAT.discontinuous_lagrange import DiscontinuousLagrange
    from FIAT.tensor_finite_element import TensorFiniteElement

    T = UFCInterval()
    P1_DG = DiscontinuousLagrange(T, 1)
    P2 = Lagrange(T, 2)

    elt = TensorFiniteElement(P1_DG, P2)
    assert elt.value_shape() == ()  # nosify
    tab = elt.tabulate(1, [(0.1, 0.2)])
    tabA = P1_DG.tabulate(1, [(0.1,)])
    tabB = P2.tabulate(1, [(0.2,)])
    for (dc, da, db) in [[(0, 0), (0,), (0,)], [(1, 0), (1,), (0,)], [(0, 1), (0,), (1,)]]:
        nose.tools.assert_almost_equal(tab[dc][0][0], tabA[da][0][0]*tabB[db][0][0])
        nose.tools.assert_almost_equal(tab[dc][1][0], tabA[da][0][0]*tabB[db][1][0])
        nose.tools.assert_almost_equal(tab[dc][2][0], tabA[da][0][0]*tabB[db][2][0])
        nose.tools.assert_almost_equal(tab[dc][3][0], tabA[da][1][0]*tabB[db][0][0])
        nose.tools.assert_almost_equal(tab[dc][4][0], tabA[da][1][0]*tabB[db][1][0])
        nose.tools.assert_almost_equal(tab[dc][5][0], tabA[da][1][0]*tabB[db][2][0])


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
    assert hdiv_elt.value_shape() == (2,)  # nosify
    assert hcurl_elt.value_shape() == (2,)  # nosify

    tabA = P1_DG.tabulate(1, [(0.1,)])
    tabB = P2.tabulate(1, [(0.2,)])

    hdiv_tab = hdiv_elt.tabulate(1, [(0.1, 0.2)])
    for (dc, da, db) in [[(0, 0), (0,), (0,)], [(1, 0), (1,), (0,)], [(0, 1), (0,), (1,)]]:
        nose.tools.assert_almost_equal(hdiv_tab[dc][0][0][0], 0.0)
        nose.tools.assert_almost_equal(hdiv_tab[dc][1][0][0], 0.0)
        nose.tools.assert_almost_equal(hdiv_tab[dc][2][0][0], 0.0)
        nose.tools.assert_almost_equal(hdiv_tab[dc][3][0][0], 0.0)
        nose.tools.assert_almost_equal(hdiv_tab[dc][4][0][0], 0.0)
        nose.tools.assert_almost_equal(hdiv_tab[dc][5][0][0], 0.0)
        nose.tools.assert_almost_equal(hdiv_tab[dc][0][1][0], tabA[da][0][0]*tabB[db][0][0])
        nose.tools.assert_almost_equal(hdiv_tab[dc][1][1][0], tabA[da][0][0]*tabB[db][1][0])
        nose.tools.assert_almost_equal(hdiv_tab[dc][2][1][0], tabA[da][0][0]*tabB[db][2][0])
        nose.tools.assert_almost_equal(hdiv_tab[dc][3][1][0], tabA[da][1][0]*tabB[db][0][0])
        nose.tools.assert_almost_equal(hdiv_tab[dc][4][1][0], tabA[da][1][0]*tabB[db][1][0])
        nose.tools.assert_almost_equal(hdiv_tab[dc][5][1][0], tabA[da][1][0]*tabB[db][2][0])

    hcurl_tab = hcurl_elt.tabulate(1, [(0.1, 0.2)])
    for (dc, da, db) in [[(0, 0), (0,), (0,)], [(1, 0), (1,), (0,)], [(0, 1), (0,), (1,)]]:
        nose.tools.assert_almost_equal(hcurl_tab[dc][0][0][0], tabA[da][0][0]*tabB[db][0][0])
        nose.tools.assert_almost_equal(hcurl_tab[dc][1][0][0], tabA[da][0][0]*tabB[db][1][0])
        nose.tools.assert_almost_equal(hcurl_tab[dc][2][0][0], tabA[da][0][0]*tabB[db][2][0])
        nose.tools.assert_almost_equal(hcurl_tab[dc][3][0][0], tabA[da][1][0]*tabB[db][0][0])
        nose.tools.assert_almost_equal(hcurl_tab[dc][4][0][0], tabA[da][1][0]*tabB[db][1][0])
        nose.tools.assert_almost_equal(hcurl_tab[dc][5][0][0], tabA[da][1][0]*tabB[db][2][0])
        nose.tools.assert_almost_equal(hcurl_tab[dc][0][1][0], 0.0)
        nose.tools.assert_almost_equal(hcurl_tab[dc][1][1][0], 0.0)
        nose.tools.assert_almost_equal(hcurl_tab[dc][2][1][0], 0.0)
        nose.tools.assert_almost_equal(hcurl_tab[dc][3][1][0], 0.0)
        nose.tools.assert_almost_equal(hcurl_tab[dc][4][1][0], 0.0)
        nose.tools.assert_almost_equal(hcurl_tab[dc][5][1][0], 0.0)


if __name__ == "__main__":
    nose.main()
