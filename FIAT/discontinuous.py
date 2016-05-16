# Copyright (C) 2014 Andrew T. T. McRae (Imperial College London)
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

from FIAT.finite_element import CiarletElement
from FIAT.dual_set import DualSet


class DiscontinuousElement(CiarletElement):
    """A copy of an existing element where all dofs are associated with the cell"""

    def __init__(self, element):
        self._element = element
        new_entity_ids = {}
        topology = element.get_reference_element().get_topology()
        for dim in sorted(topology):
            new_entity_ids[dim] = {}
            for ent in sorted(topology[dim]):
                new_entity_ids[dim][ent] = []

        new_entity_ids[dim][0] = list(range(element.space_dimension()))
        # re-initialise the dual, so entity_closure_dofs is recalculated
        self.dual = DualSet(element.dual_basis(), element.get_reference_element(), new_entity_ids)

        # fully discontinuous
        self.formdegree = element.get_reference_element().get_spatial_dimension()

    def degree(self):
        "Return the degree of the (embedding) polynomial space."
        return self._element.degree()

    def get_reference_element(self):
        "Return the reference element for the finite element."
        return self._element.get_reference_element()

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        return self._element.get_nodal_basis()

    def get_order(self):
        "Return the order of the element (may be different from the degree)"
        return self._element.get_order()

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        return self._element.get_coeffs()

    def mapping(self):
        """Return a list of appropriate mappings from the reference
        element to a physical element for each basis function of the
        finite element."""
        return self._element.mapping()

    def num_sub_elements(self):
        "Return the number of sub-elements."
        return self._element.num_sub_elements()

    def space_dimension(self):
        "Return the dimension of the finite element space."
        return self._element.space_dimension()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""
        return self._element.tabulate(order, points, entity)

    def value_shape(self):
        "Return the value shape of the finite element functions."
        return self._element.value_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        return self._element.dmats()

    def get_num_members(self, arg):
        "Return number of members of the expansion set."
        return self._element.get_num_members()
