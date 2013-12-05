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

from . import finite_element, Lagrange, dual_set

class Bubble( finite_element.FiniteElement ):
    """The Bubble finite element: the interior dofs of the Lagrange FE"""
    def __init__( self , ref_el , degree ):
        self._element = Lagrange(ref_el, degree)

        cell_dim = max(self._element.entity_dofs().keys())
        cell_entity_dofs = self._element.entity_dofs()[cell_dim][0]
        if len(cell_entity_dofs) == 0:
            raise RuntimeError('Bubble element of degree %d has no dofs' % degree)
        # 'index' is the first Lagrange node that we keep
        index = min(cell_entity_dofs)
        self.entity_ids = {}
        # Build empty entity ids
        for dim, entities in self._element.entity_dofs().iteritems():
            self.entity_ids[dim] = dict((entity, []) for entity in entities)
        # keep the IDs, starting from 'index'
        self.entity_ids[cell_dim][0] = [e - index for e in cell_entity_dofs]
        self.fsdim = len(self.entity_ids[cell_dim][0])
        # keep the dual set nodes we want, starting from 'index'
        nodes = self._element.get_dual_set().get_nodes()[index:]
        self.dual = dual_set.DualSet(nodes, self._element.get_reference_element(), self.entity_ids)

    def degree(self):
        return self._element.degree()

    def get_reference_element( self ):
        return self._element.get_reference_element()

    def get_nodal_basis( self ):
        raise NotImplementedError

    def get_dual_set( self ):
        return self.dual

    def get_order( self ):
        return 0  # Can't represent a constant function

    def dual_basis(self):
        raise NotImplementedError

    def entity_dofs(self):
        return self.entity_ids

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
        raise NotImplementedError

    def value_shape(self):
        return self._element.value_shape()

    def dmats(self):
        raise NotImplementedError

    def get_num_members(self, arg):
        raise NotImplementedError
