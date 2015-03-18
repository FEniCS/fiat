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

from __future__ import print_function

import numpy

from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import ufc_simplex
from FIAT.functional import PointEvaluation

# Tolerance for geometry identifications
epsilon = 1.e-8

def extract_unique_facet(coordinates, tolerance=epsilon):
    """Determine whether a set of points, each point described by its
    barycentric coordinates ('coordinates'), are all on one of the
    facets and return this facet."""
    facets = []
    for c in coordinates:
        on_facet = set([i for (i, l) in enumerate(c) if abs(l) < tolerance])
        facets += [on_facet]

    unique_facet = facets[0]
    for e in facets:
        unique_facet = unique_facet & e
    assert len(unique_facet) == 1, "Unable to identify unique facet "
    return unique_facet.pop()

def barycentric_coordinates(points, vertices):
    """Compute barycentric coordinates for a set of points ('points'),
    relative to a simplex defined by a set of vertices ('vertices').
    """

    # Form map matrix
    last = numpy.asarray(vertices[-1])
    T = numpy.matrix([numpy.array(v) - last for v in vertices[:-1]]).T
    detT = numpy.linalg.det(T)
    invT = numpy.linalg.inv(T)

    # Compute barycentric coordinates for all points
    coords = []
    for p in points:
        y = numpy.asarray(p) - last
        lam = invT.dot(y)
        lam = [lam[(0, i)] for i in range(len(y))]
        lam += [1.0 - sum(lam)]
        coords.append(lam)
    return coords

def map_from_reference_facet(point, vertices):
    """
    Input:
      vertices: the vertices defining the physical facet
      point: the reference point to be mapped to the facet
    """
    # Compute barycentric coordinates of point relative to reference facet:
    reference_simplex = ufc_simplex(len(vertices)-1)
    reference_vertices = reference_simplex.get_vertices()
    coords = barycentric_coordinates([point,], reference_vertices)

    # Evaluate physical coordinate of point using barycentric coordinates
    point = sum(vertices[j]*coords[0][j] for j in range(len(coords[0])))

    return tuple(point)

# FIXME: Generalise to nD
def map_to_reference_facet(points, vertices, tolerance=epsilon):
    """Given a set of points in n D and a set of vertices describing a
    facet of a simplex in n D (where the given points lie on this
    facet) map the points to the reference simplex of dimension (n-1).

    In 1D, we have that

      (x, y) = (x0, y0) + s * (x1 - x0, y1 - y0)

    So, we should have that (if x1 != x0 and/or y1!=y0)

      s = (x - x0)/(x1 - x0)
      s = (y - y0)/(y1 - y0)

    """
    print("vertices = ", vertices)
    print("points = ", points)

    # Short-hand for increased readability
    (x0, y0) = (vertices[0][0], vertices[0][1])
    (x1, y1) = (vertices[1][0], vertices[1][1])

    if abs(x1 - x0) > tolerance:
        s = [((x - x0)/(x1 - x0),) for (x, y) in points]
    else:
        s = [((y - y0)/(y1 - y0),) for (x, y) in points]
    return s

class DiscontinuousLagrangeTrace(object):
    ""
    def __init__(self, cell, k):

        tdim = cell.get_spatial_dimension()
        assert (tdim == 2 or tdim == 3)

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
        DG space dimesion on each facet times the number of facets."""
        return self.DG.space_dimension()*self.num_facets

    def entity_dofs(self):
        return self.entity_ids

    def mapping(self):
        return ["affine" for i in range(self.space_dimension())]

    def dual_basis(self):

        # First create the points
        points = []

        # For each facet, map the subcomplex DG_k dofs from the lower
        # dimensional reference element onto the facet and add to list
        # of points
        DG_k_dual_basis = self.DG.dual_basis()
        t_dim = self.cell.get_spatial_dimension()
        facets2indices = self.cell.get_topology()[t_dim - 1]

        # Iterate over the facets and add points on each facet
        for (facet, indices) in facets2indices.iteritems():
            vertices = self.cell.get_vertices_of_subcomplex(indices)
            vertices = numpy.array(vertices)
            for dof in DG_k_dual_basis:
                # PointEvaluation only carries one point
                point = dof.get_point_dict().keys()[0]
                pt = map_from_reference_facet([point,], vertices)
                points.append(pt)

        # One degree of freedom per point:
        nodes = [PointEvaluation(self.cell, x) for x in points]
        return nodes

    def tabulate(self, order, points):

        # Standard derivatives don't make sense
        assert (order == 0), "Don't know how to do derivatives"

        # Identify which facet (if any) these points are on:
        vertices = self.cell.vertices
        coordinates = barycentric_coordinates(points, vertices)
        unique_facet = extract_unique_facet(coordinates)

        # Map point to "reference facet" (facet -> interval etc)
        tdim = self.cell.get_spatial_dimension()
        facet2indices = self.cell.get_topology()[tdim - 1][unique_facet]
        vertices = self.cell.get_vertices_of_subcomplex(facet2indices)
        new_points = map_to_reference_facet(points, vertices)

        # Call self.DG.tabulate(order, new_points)
        values = self.DG.tabulate(order, new_points)
        return values

    # These functions are only needed for evaluatebasis and
    # evaluatebasisderivatives, disable those, and we should be in
    # business.
    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        msg = "Not implemented: shouldn't be implemented."
        raise Exception(msg)

    def get_num_members(self, arg):
        msg = "Not implemented: shouldn't be implemented."
        raise Exception(msg)

    def dmats(self):
        msg = "Not implemented."
        raise Exception(msg)

    def __str__(self):
        return "DiscontinuousLagrangeTrace(%s, %s)" % (self.cell, self.k)

if __name__ == "__main__":

    T = ufc_simplex(2)
    element = DiscontinuousLagrangeTrace(T, 1)
    pts = [(0.1, 0.0), (1.0, 0.0)]
    print("values = ", element.tabulate(0, pts))
    #print(element.dual_basis())

    print("\n3D ----------------")
    T = ufc_simplex(3)
    element = DiscontinuousLagrangeTrace(T, 1)
    pts = [(0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
    print(element.tabulate(0, pts))
