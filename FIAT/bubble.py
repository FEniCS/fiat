# Copyright (C) 2013 Andrew T. T. McRae (Imperial College London)
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

from __future__ import absolute_import, print_function, division

from FIAT.lagrange import Lagrange
from FIAT.restricted import RestrictedElement


class Bubble(RestrictedElement):
    """The Bubble finite element: the interior dofs of the Lagrange FE"""

    def __init__(self, ref_el, degree):
        element = Lagrange(ref_el, degree)

        cell_dim = ref_el.get_dimension()
        assert cell_dim == max(element.entity_dofs().keys())
        cell_entity_dofs = element.entity_dofs()[cell_dim][0]
        if len(cell_entity_dofs) == 0:
            raise RuntimeError('Bubble element of degree %d has no dofs' % degree)

        super(Bubble, self).__init__(element, indices=cell_entity_dofs)
