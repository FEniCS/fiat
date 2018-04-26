# -*- coding: utf-8 -*-
#
# Copyright (C) 2005-2010 Anders Logg
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

import numpy

from operator import add
from functools import partial

from FIAT.dual_set import DualSet
from FIAT.finite_element import FiniteElement


class MixedElement(FiniteElement):
    """A FIAT-like representation of a mixed element.

    :arg elements: An iterable of FIAT elements.
    :arg ref_el: The reference element (optional).

    This object offers tabulation of the concatenated basis function
    tables along with an entity_dofs dict."""
    def __init__(self, elements, ref_el=None):
        elements = tuple(elements)

        cells = set(e.get_reference_element() for e in elements)
        if ref_el is not None:
            cells.add(ref_el)
        ref_el, = cells

        # These functionals are absolutely wrong, they all map from
        # functions of the wrong shape, and potentially of different
        # shapes.  However, they are wrong precisely as FFC hacks
        # expect them to be. :(
        nodes = [L for e in elements for L in e.dual_basis()]

        entity_dofs = concatenate_entity_dofs(ref_el, elements)

        dual = DualSet(nodes, ref_el, entity_dofs)
        super(MixedElement, self).__init__(ref_el, dual, None, mapping=None)
        self._elements = elements

    def elements(self):
        return self._elements

    def num_sub_elements(self):
        return len(self._elements)

    def value_shape(self):
        return (sum(numpy.prod(e.value_shape(), dtype=int) for e in self.elements()), )

    def mapping(self):
        return [m for e in self._elements for m in e.mapping()]

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented")

    def tabulate(self, order, points, entity=None):
        """Tabulate a mixed element by appropriately splatting
        together the tabulation of the individual elements.
        """
        shape = (self.space_dimension(),) + self.value_shape() + (len(points),)

        output = {}

        sub_dims = [0] + list(e.space_dimension() for e in self.elements())
        sub_cmps = [0] + list(numpy.prod(e.value_shape(), dtype=int)
                              for e in self.elements())
        irange = numpy.cumsum(sub_dims)
        crange = numpy.cumsum(sub_cmps)

        for i, e in enumerate(self.elements()):
            table = e.tabulate(order, points, entity)

            for d, tab in table.items():
                try:
                    arr = output[d]
                except KeyError:
                    arr = numpy.zeros(shape, dtype=tab.dtype)
                    output[d] = arr

                ir = irange[i:i+2]
                cr = crange[i:i+2]
                tab = tab.reshape(ir[1] - ir[0], cr[1] - cr[0], -1)
                arr[slice(*ir), slice(*cr)] = tab

        return output

    def is_nodal(self):
        """True if primal and dual bases are orthogonal."""
        return all(e.is_nodal() for e in self._elements)


def concatenate_entity_dofs(ref_el, elements):
    """Combine the entity_dofs from a list of elements into a combined
    entity_dof containing the information for the concatenated DoFs of
    all the elements."""
    entity_dofs = {dim: {i: [] for i in entities}
                   for dim, entities in ref_el.get_topology().items()}
    offsets = numpy.cumsum([0] + list(e.space_dimension()
                                      for e in elements), dtype=int)
    for i, d in enumerate(e.entity_dofs() for e in elements):
        for dim, dofs in d.items():
            for ent, off in dofs.items():
                entity_dofs[dim][ent] += list(map(partial(add, offsets[i]), off))
    return entity_dofs
