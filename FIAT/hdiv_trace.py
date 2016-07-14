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

from __future__ import absolute_import

import numpy as np
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import ufc_simplex
from FIAT import FiniteElement
from FIAT.polynomial_set import mis


class TraceError(Exception):
    """Exception caused by tabulating a trace element on the interior of a cell,
    or the gradient of a trace element."""

    def __init__(self, msg, values, keys):

        super(TraceError, self).__init__(msg)
        self.msg = msg
        self.zeros = values
        self.D = keys


class TraceHDiv(FiniteElement):
    """Class implementing the trace of hdiv elements on general simplices."""

    def __init__(self, cell, degree):

        sd = cell.get_spatial_dimension()

        # Check to make sure spatial dim is sensible for trace
        if sd == 1:
            raise ValueError(
                "Spatial dimension not suitable for generating the trace.")

        # Otherwise, initialize some stuff and proceed
        self.cell = cell
        self.degree = degree

        # Constructing facet as a DG Lagrange element
        self.facet = ufc_simplex(sd - 1)
        self.trace = DiscontinuousLagrange(self.facet, degree)

        # Number of facets on simplex-type element
        self.num_facets = sd + 1

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
        nf = self.trace.space_dimension()

        # Filling in entity ids
        for f in range(self.num_facets):
            self._entity_dofs[sd - 1][f] = range(f * nf, (f + 1) * nf)

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.degree

    def order(self):
        """Return the order of the trace element."""
        return self.degree

    def get_formdegree(self):
        """Returns the form degree of the facet element (FEEC)"""
        return self.trace.get_formdegree()

    def dual_basis(self):
        """Returns the dual basis corresponding to a single facet element.
        Note: that this is not the dual set of the trace element."""
        return self.trace.dual_basis()

    def mapping(self):
        """Returns the mapping from the reference
        element to a trace element."""
        return self.trace.mapping()

    def space_dimension(self):
        "Return the dimension of the trace finite element space."
        return self.trace.space_dimension() * self.num_facets

    def get_reference_element(self):
        "Return the reference element where the traces are defined on."
        return self.cell

    def entity_dofs(self):
        """Return the entity dictionary."""
        return self._entity_dofs

    def entity_closure_dofs(self):
        """Return the entity closure dictionary."""
        # They are the same as entity_dofs for the trace element
        return self._entity_dofs

    def tabulate(self, order, points, entity):
        """Return tabulated values of basis functions at given points."""

        facet_dim = self.cell.get_spatial_dimension() - 1
        sdim = self.space_dimension()

        # Initializing dictionary with zeros
        phivals = {}
        for i in range(order + 1):
            alphas = mis(self.cell.get_spatial_dimension(), i)
            for alpha in alphas:
                phivals[alpha] = np.zeros(shape=(sdim, len(points)))
        key = phivals.keys()

        # If doing cell-wise tabulation, raise TraceError and return zeros
        if (entity is None) or entity[0] != facet_dim:
            raise TraceError("Trace elements can only be tabulated on facet entities.",
                             phivals, key)

        # Retrieve function evaluations (order = 0 case)
        nf = self.trace.space_dimension()
        facet_id = entity[1]
        nonzerovals = self.trace.tabulate(0, points).values()[0]
        phivals[key[-1]][nf*facet_id:nf*(facet_id + 1), :] = nonzerovals

        # If asking for gradient evaluations, raise TraceError
        # but return functon evaluations, and zeros for the gradient.
        if order > 0:
            raise TraceError("No gradient evaluations on trace elements.",
                             phivals, key)

        return phivals

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return self.trace.value_shape()
