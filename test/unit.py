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

    for i in range(26):
        random.seed(42)
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


if __name__ == "__main__":
    nose.main(defaultTest='unit')
