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

from finite_element import FiniteElement
from dual_set import DualSet


class Interiorize(FiniteElement):
    """An element produced from an existing element by only keeping degrees of
    freedom associated with the cell"""

    def __init__(self, element):
        self._element = element
        self.order = 0  # can't (in general) represent constant function
        self.ref_el = element.get_reference_element()
        sd = self.ref_el.get_spatial_dimension()
        self.formdegree = sd  # fully discontinuous
        self._mapping = element.mapping()[0]
        
        # make entity dof list of new element.
        # also build a 'mapping' from new dofs to old dofs.
        new_entity_dofs = {}
        self.dofmapping = []
        index = 0
        old_dofs = element.entity_dofs()
        sorted_old_dofs = sorted(old_dofs)
        for dim in sorted_old_dofs:
            new_entity_dofs[dim] = {}
            for ent in old_dofs[dim]:
                # discard dofs if not belonging to cell
                if dim != sorted_old_dofs[-1]:
                    new_entity_dofs[dim][ent] = []
                else:
                    # keep dofs, and add to dofmapping
                    new_entity_dofs[dim][ent] = []
                    for foo in old_dofs[dim][ent]:
                        self.dofmapping.append(foo)
                        new_entity_dofs[dim][ent].append(index)
                        index += 1

        self.entity_ids = new_entity_dofs
        self.fsdim = index

        # set up dual basis
        nodes = []
        for i in self.dofmapping:
            nodes.append(element.dual_basis()[i])
        self.dual = DualSet(nodes, self.ref_el, self.entity_ids)

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self._element.degree()

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented")

    def num_sub_elements(self):
        "Return the number of sub-elements."
        return 1

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        return self.fsdim

    def tabulate(self, order, points):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""
        old_tabulate = self._element.tabulate(order, points)
        tempdict = {}
        for deriv in old_tabulate:
            tempdict[deriv] = old_tabulate[deriv][self.dofmapping]
        return tempdict

    def value_shape(self):
        "Return the value shape of the finite element functions."
        return self._element.value_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented")
