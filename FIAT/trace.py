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

from __future__ import absolute_import
from __future__ import print_function

import numpy

from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import ufc_simplex
from FIAT.functional import PointEvaluation
from FIAT.polynomial_set import mis

# Tolerance for geometry identifications
epsilon = 1.e-8


def extract_unique_facet(coordinates, tolerance=epsilon):
    """Determine whether a set of points, each point described by its
    barycentric coordinates ('coordinates'), are all on one of the
    facets and return this facet and whether search has been
    successful.
    """
    facets = []
    for c in coordinates:
        on_facet = set([i for (i, l) in enumerate(c) if abs(l) < tolerance])
        facets += [on_facet]

    unique_facet = facets[0]
    for e in facets:
        unique_facet = unique_facet & e

    # Handle coordinates not on facets somewhat gracefully
    if (len(unique_facet) != 1):
        return (None, False)

    # If we have a unique facet, return it and success
    return (unique_facet.pop(), True)


def barycentric_coordinates(points, vertices):
    """Compute barycentric coordinates for a set of points ('points'),
    relative to a simplex defined by a set of vertices ('vertices').
    """

    # Form map matrix
    last = numpy.asarray(vertices[-1])
    T = numpy.matrix([numpy.array(v) - last for v in vertices[:-1]]).T
    invT = numpy.linalg.inv(T)

    # Compute barycentric coordinates for all points
    coords = []
    for p in points:
        y = numpy.asarray(p) - last
        lam = invT.dot(y.T)
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
    reference_simplex = ufc_simplex(len(vertices) - 1)
    reference_vertices = reference_simplex.get_vertices()
    coords = barycentric_coordinates([point, ], reference_vertices)[0]

    # Evaluate physical coordinate of point using barycentric coordinates
    point = sum(vertices[j] * coords[j] for j in range(len(coords)))

    return tuple(point)


def map_to_reference_facet(points, vertices, facet):
    """Given a set of points in n D and a set of vertices describing a
    facet of a simplex in n D (where the given points lie on this
    facet) map the points to the reference simplex of dimension (n-1).
    """

    # Compute barycentric coordinates of points with respect to
    # the full physical simplex
    all_coords = barycentric_coordinates(points, vertices)

    # Extract vertices of reference facet simplex
    reference_facet_simplex = ufc_simplex(len(vertices) - 2)
    ref_vertices = reference_facet_simplex.get_vertices()

    reference_points = []
    for (i, coords) in enumerate(all_coords):
        # Extract correct subset of barycentric coordinates since we
        # know which facet we are on
        new_coords = [coords[j] for j in range(len(coords)) if (j != facet)]

        # Evaluate reference coordinate of point using revised
        # barycentric coordinates
        reference_pt = sum(numpy.asarray(ref_vertices[j]) * new_coords[j]
                           for j in range(len(new_coords)))

        reference_points += [reference_pt]
    return reference_points


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
        for dim, entities in topology.items():
            self.entity_ids[dim] = {}
            for entity in entities:
                self.entity_ids[dim][entity] = {}

        # For each facet, we have dim(DG_k on that facet) number of dofs
        n = self.DG.space_dimension()
        for i in range(self.num_facets):
            self.entity_ids[tdim - 1][i] = range(i * n, (i + 1) * n)

    def degree(self):
        return self.k

    def value_shape(self):
        return ()

    def space_dimension(self):
        """The space dimension of the trace space corresponds to the
        DG space dimesion on each facet times the number of facets."""
        return self.DG.space_dimension() * self.num_facets

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
        for (facet, indices) in facets2indices.items():
            vertices = self.cell.get_vertices_of_subcomplex(indices)
            vertices = numpy.array(vertices)
            for dof in DG_k_dual_basis:
                # PointEvaluation only carries one point
                point = list(dof.get_point_dict().keys())[0]
                pt = map_from_reference_facet([point, ], vertices)
                points.append(pt)

        # One degree of freedom per point:
        nodes = [PointEvaluation(self.cell, x) for x in points]
        return nodes

    def tabulate(self, order, points):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""

        # Standard derivatives don't make sense, but return zero
        # because mixed elements compute all derivatives at once
        if (order > 0):
            values = {}
            sdim = self.space_dimension()
            alphas = mis(self.cell.get_spatial_dimension(), order)
            for alpha in alphas:
                values[alpha] = numpy.zeros(shape=(sdim, len(points)))
            return values

        # Identify which facet (if any) these points are on:
        vertices = self.cell.vertices
        coordinates = barycentric_coordinates(points, vertices)
        (unique_facet, success) = extract_unique_facet(coordinates)

        # All other basis functions evaluate to zero, so create an
        # array of the right size
        sdim = self.space_dimension()
        values = numpy.zeros(shape=(sdim, len(points)))

        # ... and plug in the non-zero values in just the right place
        # if we found a unique facet
        if success:

            # Map point to "reference facet" (facet -> interval etc)
            new_points = map_to_reference_facet(points, vertices, unique_facet)

            # Call self.DG.tabulate(order, new_points) to compute the
            # values of the points for the degrees of freedom on this facet
            non_zeros = list(self.DG.tabulate(order, new_points).values())[0]
            m = non_zeros.shape[0]
            dg_dim = self.DG.space_dimension()
            values[dg_dim*unique_facet:dg_dim*unique_facet + m, :] = non_zeros

        # Return expected dictionary
        tdim = self.cell.get_spatial_dimension()
        key = tuple(0 for i in range(tdim))

        return {key: values}

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
