# Copyright (C) 2012-2015 Marie E. Rognes and David A. Ham
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

from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import ufc_simplex

class TraceSpace(object):
    def __init__(self, cell, k):

        # Store input cell and polynomial degree (k)
        self.cell = cell
        self.k = k

        # Create DG_k space on the facet(s) of the cell
        tdim = self.cell.get_spatial_dimension()
        self.facet = ufc_simplex(tdim - 1)
        self.DG = DiscontinuousLagrange(self.facet, k)

        # Count number of facets for given cell. Assumption: we are on
        # simplices
        self.num_facets = tdim + 1

        # Construct entity ids. Initialize all to empty, will fill
        # later.
        self.entity_ids = {}
        topology = cell.get_topology()
        for dim, entities in topology.iteritems():
            self.entity_ids[dim] = {}
            for entity in entities:
                self.entity_ids[dim][entity] = {}

        # For each facet, we have dim(DG_k on that facet) number of dofs
        n = self.DG.space_dimension()
        for i in range(self.num_facets):
            self.entity_ids[tdim-1][i] = range(i*n, (i+1)*n)

    def degree(self):
        return self.k

    def value_shape(self):
        return ()

    def space_dimension(self):
        """The space dimension of the trace space corresponds to the
        DG space on each facet."""
        return self.DG.space_dimension()*self.num_facets

    def entity_dofs(self):
        return self.entity_ids

    def mapping(self):
        return ["affine" for i in self.space_dimension()]

    def dual_basis(self):

        # I think this is supposed to be a list, each element of the
        # list corresponds to a representation of a degree of freedom
        # as a linear combination

        # For each facet, map DG_k on reference facet to this facet,
        # add node as PointEvaluation on this point. Something like
        # this:

        #nodes = [functional.PointEvaluation( self.cell , x )
        #             for x in points ]

        nodes = []
        return nodes

    def tabulate(self, order, points):

        # Standard derivatives don't make sense (cf manifolds
        # work). Maybe derivatives are not needed?
        assert (order == 0),  "Don't know how to do derivatives"

        # Check that points are on edge

        # Identify which edge

        # Map point to "reference facet" (edge -> interval etc)

        # Call self.DG.tabulate(order, new_points)

        pass

    # These functions are only needed for evaluatebasis and
    # evaluatebasisderivatives, disable those, and you should be in
    # business.
    def get_num_members(self, arg):
        raise Exception, "Not implemented: shouldn't be implemented."

    def dmats(self):
        raise Exception, """Not implemented: don't know how to implement, but that's ok."""

if __name__ == "__main__":
    T = ufc_simplex(2)
    element = TraceSpace(T, 1)
