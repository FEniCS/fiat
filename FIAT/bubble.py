# Copyright (C) 2013 Andrew T. T. McRae (Imperial College London)
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

from .finite_element import FiniteElement
from . import Lagrange, dual_set
from six import iteritems


class Bubble(FiniteElement):
    """The Bubble finite element: the interior dofs of the Lagrange FE"""

    def __init__(self, ref_el, degree):
        self._element = Lagrange(ref_el, degree)

        cell_dim = max(self._element.entity_dofs().keys())
        cell_entity_dofs = self._element.entity_dofs()[cell_dim][0]
        if len(cell_entity_dofs) == 0:
            raise RuntimeError('Bubble element of degree %d has no dofs' % degree)
        # 'index' is the first Lagrange node that we keep
        self.first_node_index = min(cell_entity_dofs)
        entity_ids = {}
        # Build empty entity ids
        for dim, entities in iteritems(self._element.entity_dofs()):
            entity_ids[dim] = dict((entity, []) for entity in entities)
        # keep the IDs, starting from 'index'
        entity_ids[cell_dim][0] = [e - self.first_node_index for e in cell_entity_dofs]
        self.fsdim = len(entity_ids[cell_dim][0])
        # keep the dual set nodes we want, starting from 'index'
        nodes = self._element.dual_basis()[self.first_node_index:]
        self.dual = dual_set.DualSet(nodes, ref_el, entity_ids)

    def degree(self):
        return self._element.degree()

    def get_reference_element(self):
        return self._element.get_reference_element()

    def get_nodal_basis(self):
        raise NotImplementedError

    def get_order(self):
        return 0  # Can't represent a constant function

    def get_coeffs(self):
        raise NotImplementedError

    def get_formdegree(self):
        return self._element.get_formdegree()

    def mapping(self):
        return [self._element._mapping]*self.space_dimension()

    def num_sub_elements(self):
        raise NotImplementedError

    def space_dimension(self):
        return self.fsdim

    def tabulate(self, order, points):
        return dict((k, v[self.first_node_index:, :])
                    for k, v in self._element.tabulate(order, points).items())

    def value_shape(self):
        return self._element.value_shape()

    def dmats(self):
        raise NotImplementedError

    def get_num_members(self, arg):
        raise NotImplementedError
