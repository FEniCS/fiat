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

from __future__ import absolute_import, print_function, division

import numpy as np
from FIAT.discontinuous_lagrange import DiscontinuousLagrange
from FIAT.reference_element import ufc_simplex
from FIAT.functional import PointEvaluation
from FIAT.polynomial_set import mis
from FIAT import FiniteElement
from FIAT import dual_set

# Numerical tolerance for facet-entity identifications
epsilon = 1e-10


class TraceError(Exception):
    """Exception caused by tabulating a trace element on the interior of a cell,
    or the gradient of a trace element."""

    def __init__(self, msg):
        super(TraceError, self).__init__(msg)
        self.msg = msg


class HDivTrace(FiniteElement):
    """Class implementing the trace of hdiv elements."""

    def __init__(self, ref_el, degree):
        sd = ref_el.get_spatial_dimension()
        if sd in (0, 1):
            raise ValueError("Cannot use this trace class on a %d-dimensional cell.")
        self.ref_el = ref_el
        self.polydegree = degree

        # Constructing facet element as a discontinuous Lagrange element
        self.dclagrange = DiscontinuousLagrange(ufc_simplex(sd - 1), degree)

        # Construct entity ids (assigning top. dim. and initializing as empty)
        self._entity_dofs = {}

        # Looping over dictionary of cell topology to construct the empty
        # dictionary for entity ids of the trace element
        topology = self.ref_el.get_topology()
        for top_dim, entities in topology.items():
            self._entity_dofs[top_dim] = {}
            for entity in entities:
                self._entity_dofs[top_dim][entity] = []

        # Filling in entity ids and generating points for dual basis
        nf = self.dclagrange.space_dimension()
        points = []
        self.num_facets = sd + 1
        for f in range(self.num_facets):
            self._entity_dofs[sd - 1][f] = range(f * nf, (f + 1) * nf)

            for dof in self.dclagrange.dual_basis():
                facet_point = list(dof.get_point_dict().keys())[0]
                transform = self.ref_el.get_entity_transform(sd - 1, f)
                points.append(tuple(transform(facet_point)))

        # Setting up dual basis - only point evaluations
        nodes = [PointEvaluation(self.ref_el, pt) for pt in points]
        self._dual = dual_set.DualSet(nodes, self.ref_el, self._entity_dofs)

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polydegree

    def order(self):
        """Return the order of the trace element."""
        return self.polydegree

    def get_formdegree(self):
        """Returns the form degree of the facet element (FEEC)"""
        return self.dclagrange.get_formdegree()

    def dual_basis(self):
        """Returns the dual basis corresponding to a single facet element.
        Note: that this is not the dual set of the trace element."""
        return self._dual.get_nodes()

    def mapping(self):
        """Returns the mapping from the reference
        element to a trace element."""
        return self.dclagrange.mapping()

    def space_dimension(self):
        "Return the dimension of the finite element space."
        return self.dclagrange.space_dimension() * self.num_facets

    def get_reference_element(self):
        "Return the reference element where the traces are defined on."
        return self.ref_el

    def entity_dofs(self):
        """Return the entity dictionary."""
        return self._dual.get_entity_ids()

    def entity_closure_dofs(self):
        """Return the entity closure dictionary."""
        return self._dual.get_entity_closure_ids()

    def get_dual_set(self):
        "Return the dual for the finite element."
        return self._dual

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented for the trace element.")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented for the trace element.")

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of basis functions at given points."""

        facet_dim = self.ref_el.get_spatial_dimension() - 1
        sdim = self.space_dimension()
        nf = self.dclagrange.space_dimension()

        # Initializing dictionary with zeros
        phivals = {}
        for i in range(order + 1):
            alphas = mis(self.ref_el.get_spatial_dimension(), i)
            for alpha in alphas:
                phivals[alpha] = np.zeros(shape=(sdim, len(points)))
        key = phivals.keys()
        evalkey = list(key)[-1]

        # If entity is None, identify facet using numerical tolerance and
        # return the tabulated values
        if entity is None:

            # Attempt to identify which facet (if any) the given points are on
            vertices = self.ref_el.vertices
            coordinates = barycentric_coordinates(points, vertices)
            (unique_facet, success) = extract_unique_facet(coordinates)

            # If we are successful in finding a unique facet, we fill in the non-zero values
            if success:

                # Map points to the reference facet
                new_points = map_to_reference_facet(points, vertices, unique_facet)

                # Retrieve values by tabulating the DiscontinuousLagrange element
                nonzerovals = list(self.dclagrange.tabulate(order, new_points).values())[0]
                phivals[evalkey][nf*unique_facet:nf*(unique_facet + 1), :] = nonzerovals

                return phivals

        # If the user is directly specifying cell-wise tabulation, return TraceError
        elif entity[0] != facet_dim:
            for key in phivals.keys():
                phivals[key] = TraceError("Trace elements can only be tabulated on facet entities.")
            return phivals
        else:
            # Retrieve function evaluations (order = 0 case)
            facet_id = entity[1]
            nonzerovals = list(self.dclagrange.tabulate(0, points).values())[0]
            phivals[evalkey][nf*facet_id:nf*(facet_id + 1), :] = nonzerovals

            # If asking for gradient evaluations, insert TraceError in gradient evaluations
            # but return functon evaluations
            if order > 0:
                for key in phivals.keys():
                    if key != evalkey:
                        phivals[key] = TraceError("Gradient evaluations are illegal on trace elements.")
            return phivals

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return self.dclagrange.value_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented for the trace element.")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented for the trace element.")


# The following functions are credited to Marie E. Rognes:
def extract_unique_facet(coordinates, tolerance=epsilon):
    """Determines whether a set of points (described in its barycentric coordinates)
    are all on one of the facet sub-entities, and return the particular facet and
    whether the search has been successful.

    :arg coordinates: A set of points described in barycentric coordinates.
    :arg tolerance: A fixed tolerance for geometric identifications.
    """
    facets = []
    for c in coordinates:
        on_facet = set([i for (i, l) in enumerate(c) if abs(l) < tolerance])
        facets += [on_facet]

    unique_facet = facets[0]
    for f in facets:
        unique_facet = unique_facet & f

    # Handle coordinates not on facets
    if len(unique_facet) != 1:
        return (None, False)

    # If we have a unique facet, return it and success
    return (unique_facet.pop(), True)


def barycentric_coordinates(points, vertices):
    """Computes the barycentric coordinates for a set of points relative to a
    simplex defined by a set of vertices.

    :arg points: A set of points.
    :arg vertices: A set of vertices that define the simplex.
    """

    # Form mapping matrix
    last = np.asarray(vertices[-1])
    T = np.matrix([np.array(v) - last for v in vertices[:-1]]).T
    invT = np.linalg.inv(T)

    # Compute the barycentric coordinates for all points
    coords = []
    for p in points:
        y = np.asarray(p) - last
        bary = invT.dot(y.T)
        bary = [bary[(0, i)] for i in range(len(y))]
        bary += [1.0 - sum(bary)]
        coords.append(bary)
    return coords


def map_from_reference_facet(point, vertices):
    """Evaluates the physical coordinate of a point using barycentric
    coordinates.

    :arg point: The reference points to be mapped to the facet.
    :arg vertices: The vertices defining the physical element.
    """

    # Compute the barycentric coordinates of the point relative to the reference facet
    reference_simplex = ufc_simplex(len(vertices) - 1)
    reference_vertices = reference_simplex.get_vertices()
    coords = barycentric_coordinates([point, ], reference_vertices)[0]

    # Evaluates the physical coordinate of the point using barycentric coordinates
    point = sum(vertices[j] * coords[j] for j in range(len(coords)))
    return tuple(point)


def map_to_reference_facet(points, vertices, facet):
    """Given a set of points and vertices describing a facet of a simplex in n-dimensional
    coordinates (where the points lie on the facet), map the points to the reference simplex
    of dimension (n-1).

    :arg points: A set of points in n-D.
    :arg vertices: A set of vertices describing a facet of a simplex in n-D.
    :arg facet: Integer representing the facet number.
    """

    # Compute the barycentric coordinates of the points with respect to the
    # full physical simplex
    all_coords = barycentric_coordinates(points, vertices)

    # Extract vertices of the reference facet
    reference_facet_simplex = ufc_simplex(len(vertices) - 2)
    reference_vertices = reference_facet_simplex.get_vertices()

    reference_points = []
    for (i, coords) in enumerate(all_coords):
        # Extract the correct subset of barycentric coordinates since we know
        # which facet we are on
        new_coords = [coords[j] for j in range(len(coords)) if j != facet]

        # Evaluate the reference coordinate of a point in barycentric coordinates
        reference_pt = sum(np.asarray(reference_vertices[j]) * new_coords[j]
                           for j in range(len(new_coords)))

        reference_points += [reference_pt]
    return reference_points
