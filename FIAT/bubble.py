# Copyright (C) 2013 Andrew T. T. McRae (Imperial College London)
# Copyright (C) 2015 Jan Blechta
# Copyright (C) 2018 Patrick E. Farrell
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

from FIAT.lagrange import Lagrange
from FIAT.restricted import RestrictedElement
from itertools import chain


class CodimBubble(RestrictedElement):
    """Bubbles of a certain codimension."""

    def __init__(self, ref_el, degree, codim):
        element = Lagrange(ref_el, degree)

        cell_dim = ref_el.get_dimension()
        assert cell_dim == max(element.entity_dofs().keys())
        dofs = list(sorted(chain(*element.entity_dofs()[cell_dim - codim].values())))
        if len(dofs) == 0:
            raise RuntimeError('Bubble element of degree %d and codimension %d has no dofs' % (degree, codim))

        super(CodimBubble, self).__init__(element, indices=dofs)


class Bubble(CodimBubble):
    """The bubble finite element: the dofs of the Lagrange FE in the interior of the cell"""

    def __init__(self, ref_el, degree):
        super(Bubble, self).__init__(ref_el, degree, codim=0)


class FacetBubble(CodimBubble):
    """The facet bubble finite element: the dofs of the Lagrange FE in the interior of the facets"""

    def __init__(self, ref_el, degree):
        super(FacetBubble, self).__init__(ref_el, degree, codim=1)
