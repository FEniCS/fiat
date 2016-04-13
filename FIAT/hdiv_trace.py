# Copyright (C) 2016 Thomas H. Gibson
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

import numpy as np
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import ufc_simplex
from FIAT import FiniteElement


class TraceError(Exception):

    """Exception caused by tabulating a trace element on the interior of a cell."""

    def __init__(self, msg, zeros):

        super(TraceError, self).__init__(msg)
        self.zeros = zeros


class TraceHDiv(FiniteElement):

    """Class implementing the trace of hdiv elements."""

    def __init__(self, cell, polyDegree):

        spaceDim = cell.get_spatial_dimension()

        # Check to make sure spatial dim is sensible for trace
        if spaceDim == 1:
            raise ValueError(
                "Spatial dimension not suitable for generating the trace.")

        # Otherwise, initialize some neat stuff and proceed
        self.cell = cell
        self.polyDegree = polyDegree

        # Constructing facet as a DC Lagrange element
        self.facet = ufc_simplex(spaceDim - 1)
        self.DCLagrange = DiscontinuousLagrange(self.facet, polyDegree)

        # Number of facets on simplex-type element
        self.num_facets = spaceDim + 1

        # Construct entity ids (assigning top. dim. and initializing as empty)
        self._entity_dofs = {}

        # Looping over dictionary of cell topology to construct the empty
        # dictionary for entity ids of the trace element
        topology = cell.get_topology()

        for top_dim, entities in topology.items():
            self._entity_dofs[top_dim] = {}

            for entity in entities:
                self._entity_dofs[top_dim][entity] = []

        # For each facet, we have nf = dim(facet) number of dofs
        # In this case, the facet is a DCLagrange element
        nf = self.DCLagrange.space_dimension()

        # Filling in entity ids
        for f in range(self.num_facets):
            self._entity_dofs[spaceDim - 1][f] = range(f * nf, (f + 1) * nf)

# Compute the nodes on the closure of facet.
#        self._entity_closure_dofs = {}
#        for dim, entities in cell.sub_entities.iteritems():
#            self._entity_closure_dofs[dim] = {}
#
#            for e, sub_entities in entities.iteritems():
#                ids = []
#
#                for d, se in sub_entities:
#                    ids += self._entity_dofs[d][se]
#                self._entity_closure_dofs[d][e] = ids

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polyDegree

    def space_dimension(self):
        "Return the dimension of the trace finite element space."
        return self.DCLagrange.space_dimension() * self.num_facets

    def get_reference_element(self):
        "Return the reference element for the trace element."
        return self.facet

    def entity_dofs(self):
        """Return the entity dictionary."""
        return self._entity_dofs

    def entity_closure_dofs(self):
        """Return the entity closure dictionary."""
        # They are the same as entity_dofs for the trace element
        return self._entity_dofs

    def tabulate(self, order, points, entity):
        """Return tabulated values basis functions at given points."""

        facet_dim = self.cell.get_spatial_dimension() - 1
        phiVals = np.zeros((self.space_dimension(), len(points)))

        key = tuple(0 for i in range(facet_dim + 1))

        # No derivatives
        if (order > 0):
            zeros = {key: phiVals}
            raise TraceError(
                "Only allowed for function evaluations - No derivatives.", zeros)

        if (entity is None) or entity[0] != facet_dim:
            zeros = {key: phiVals}
            raise TraceError(
                "Trace elements can only be tabulated on facets.",
                zeros)
        else:
            # Initialize basis function values at nodes to be 0 since
            # all basis functions are 0 except for specific phi on a facet
            nf = self.DCLagrange.space_dimension()
            facet_id = entity[1]

            # Tabulate basis function values on specific facet
            nonzeroVals = self.DCLagrange.tabulate(order, points).values()[0]
            phiVals[nf * facet_id:nf * (facet_id + 1), :] = nonzeroVals
            return {key: phiVals}

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return self.DCLagrange.value_shape()
