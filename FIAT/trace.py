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

import numpy

from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import ufc_simplex
from FIAT.functional import PointEvaluation

class TraceElement(object):
    def __init__(self, cell, k):

        # Only support 2D first
        tdim = cell.get_spatial_dimension()
        assert tdim == 2, "Only trace elements on triangles supported for now"

        # Store input cell and polynomial degree (k)
        self.cell = cell
        self.k = k

        # Create DG_k space on the facet(s) of the cell
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

        # First create the points
        points = []

        # For each facet, map the subcomplex DG_k dofs from the lower
        # dimensional reference element onto the facet and add to list
        # of points

        # FIXME: Generalise to nD
        def map_from_reference_facet(point, vertices):
            pt = vertices[0] + point[0]*(vertices[1] - vertices[0])
            print "pt = ", pt
            return tuple(pt)

        DG_k_dual_basis = self.DG.dual_basis()
        t_dim = self.cell.get_spatial_dimension()
        facets2indices = self.cell.get_topology()[t_dim - 1]
        for (facet, indices) in facets2indices.iteritems():
            vertices = self.cell.get_vertices_of_subcomplex(indices)
            vertices = numpy.array(vertices)
            for dof in DG_k_dual_basis:
                # PointEvaluation only carries one point
                point = dof.get_point_dict().keys()[0]
                pt = map_from_reference_facet(point, vertices)
                points.append(pt)

        # One degree of freedom per point:
        nodes = [PointEvaluation(self.cell, x) for x in points]

        return nodes

    def tabulate(self, order, points):

        # Standard derivatives don't make sense
        assert (order == 0), "Don't know how to do derivatives"

        # Check that points are on edge

        # Identify which edge

        # Map point to "reference facet" (edge -> interval etc)

        # Call self.DG.tabulate(order, new_points)
        # FIXME:
        return self.DG.tabulate(order, [(0.0,) for i in points])

    # These functions are only needed for evaluatebasis and
    # evaluatebasisderivatives, disable those, and we should be in
    # business.
    def get_num_members(self, arg):
        msg = "Not implemented: shouldn't be implemented."
        raise Exception(msg)

    def dmats(self):
        msg = "Not implemented."
        raise Exception(msg)

    def __str__(self):
        return "TraceElement(%s, %s)" % (self.cell, self.k)

if __name__ == "__main__":

    print "-"*80
    T = ufc_simplex(2)
    element = TraceElement(T, 1)
    print element.entity_ids
    print element.dual_basis()
    pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    print element.tabulate(0, pts)

    #print "-"*80
    #T = ufc_simplex(3)
    #element = TraceElement(T, 1)
    #print element
    #print element.dual_basis()
