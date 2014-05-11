# Copyright (C) 2014 Imperial College London and others.
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
# Written by David A. Ham (david.ham@imperial.ac.uk)
import numpy
import quadrature as quad
from reference_element import LINE, TRIANGLE, TETRAHEDRON, ReferenceElement


class SimplexFacetQuadratureRule(object):
    """A class which wraps quadrature rules to provide quadrature on the
    facets of an element."""
    def __init__(self, ref_el, m):

        assert ref_el.shape in (LINE, TRIANGLE, TETRAHEDRON)

        self.ref_el = ref_el
        self.facet_ref_el = ReferenceElement(ref_el.shape - 1)
        self.quad = facet_quadrature

    def get_points(self, facet):

        if self.ref_el.shape in (TRIANGLE, TETRAHEDRON):

            facetpts = numpy.array(self.quad.get_points())

            cellpts = numpy.zeros((facetpts.shape[0],
                                   facetpts.shape[1]+1))

            if facet == 0:
                # The diagonal facet. Coordinates on this facet sum to 1.
                cellpts[:, :-1] = pts
                cellpts[:, -1] = 1. - cellpts.sum(0)

            else:
                # Axis-aligned facets. Points on this facet have the
                # corresponding entry zeroed.
                cellpts[:, :facet-1] = pts[:, :facet-1]
                cellpts[:, facet:] = pts[:, facet-1:]

            return cellpts

        else:
            # Line case
            return np.array([[float(facet)]])

    def get_weights(self):

        if self.ref_el.shape in (TRIANGLE, TETRAHEDRON):
            return self.quad.get_weights()
        else:
            return [1.]
