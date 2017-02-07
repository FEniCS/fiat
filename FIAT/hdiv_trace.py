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
from six import iteritems

import numpy as np
from FIAT.reference_element import (ufc_simplex, Point, Simplex,
                                    TensorProductCell,
                                    FiredrakeQuadrilateral)
from FIAT.functional import PointEvaluation
from FIAT.polynomial_set import mis
from FIAT import (FiniteElement, dual_set,
                  DiscontinuousLagrange,
                  TensorProductElement)

# Numerical tolerance for facet-entity identifications
epsilon = 1e-10


class TraceError(Exception):
    """Exception caused by tabulating a trace element on the interior of a cell,
    or the gradient of a trace element."""

    def __init__(self, msg):
        super(TraceError, self).__init__(msg)
        self.msg = msg


class HDivTrace(FiniteElement):
    """Class implementing the trace of hdiv elements. This class
    is a stand-alone element family that produces a DG-facet field.
    This element is what's produced after performing the trace
    operation on an existing H(Div) element.

    This element is also known as the discontinuous trace field that
    arises in several DG formulations.
    """

    def __init__(self, ref_el, degree):
        """Constructor for the HDivTrace element.

        :arg ref_el: a reference element, which may be a tensor product
                     cell.
        :arg degree: the degree of approximation.
        """
        sd = ref_el.get_spatial_dimension()
        if sd in (0, 1):
            raise ValueError("Cannot take the trace of a %d-dim cell." % sd)

        facet_sd = sd - 1
        dg_elements = {}
        entity_dofs = {}
        topology = ref_el.get_topology()
        for top_dim, entities in iteritems(topology):
            cell = ref_el.construct_subelement(top_dim)
            entity_dofs[top_dim] = {}

            if cell.get_spatial_dimension() == facet_sd:
                dg_elements[top_dim] = construct_dg_element(cell, degree)

            for entity in entities:
                entity_dofs[top_dim][entity] = []

        offset = 0
        pts = []
        for facet_dim in dg_elements:
            element = dg_elements[facet_dim]
            nf = element.space_dimension()
            num_facets = len(topology[facet_dim])

            for i in range(num_facets):
                entity_dofs[facet_dim][i] = range(offset + i * nf,
                                                  offset + (i + 1) * nf)
                for dof in element.dual_basis():
                    facet_pt = list(dof.get_point_dict().keys())[0]
                    transform = ref_el.get_entity_transform(facet_dim, i)
                    pts.append(tuple(transform(facet_pt)))

            offset += entity_dofs[facet_dim][num_facets - 1][-1] + 1

        nodes = [PointEvaluation(ref_el, pt) for pt in pts]
        dual = dual_set.DualSet(nodes, ref_el, entity_dofs)

        super(HDivTrace, self).__init__(ref_el, dual, order=degree,
                                        formdegree=facet_sd,
                                        mapping="affine")
        self.dg_elements = dg_elements
        self.polydegree = degree

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polydegree

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented for the trace element.")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented for the trace element.")

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to a given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     tabulated values are computed by geometrically
                     approximating which facet the points are on.

        .. note ::

        Performing illegal tabulations on this element will result in either
        a tabulation table of `numpy.nan` arrays (`entity=None` case), or
        insertions of the `TraceError` exception class. This is due to the
        fact that performing cell-wise tabulations, or asking for any order
        of derivative evaluations, are not mathematically well-defined.
        """
        sd = self.ref_el.get_spatial_dimension()
        facet_sd = sd - 1
        phivals = {}
        for i in range(order + 1):
            alphas = mis(sd, i)

            for alpha in alphas:
                phivals[alpha] = np.zeros(shape=(self.space_dimension(),
                                                 len(points)))

        evalkey = (0,) * sd

        if entity is None:
            if not isinstance(self.ref_el, Simplex):
                raise NotImplementedError(
                    "Tabulating this element on a %s cell without providing "
                    "an entity is not currently supported." % type(self.ref_el)
                )

            vertices = self.ref_el.vertices
            coordinates = barycentric_coordinates(points, vertices)
            unique_facet, success = extract_unique_facet(coordinates)

            if success:
                new_points = map_to_reference_facet(points, vertices, unique_facet)
                element = self.dg_elements[facet_sd]
                nf = element.space_dimension()
                nonzerovals = list(element.tabulate(order, new_points).values())[0]

                phivals[evalkey][nf*unique_facet:nf * (unique_facet + 1), :] = nonzerovals

            else:
                for key in phivals:
                    phivals[key] = np.full(shape=(sd, len(points)), fill_value=np.nan)

            return phivals

        entity_dim, entity_id = entity

        if entity_dim not in self.dg_elements:
            for key in phivals:
                msg = "Tabulating the HDivTrace element is only allowed on facet entities"
                phivals[key] = TraceError(msg)

        else:
            element = self.dg_elements[entity_dim]
            nf = element.space_dimension()
            nonzerovals = list(element.tabulate(0, points).values())[0]

            phivals[evalkey][nf*entity_id:nf * (entity_id + 1), :] = nonzerovals

            if order > 0:
                msg = "Gradients on trace elements are not well-defined"
                for key in phivals:
                    if key != evalkey:
                        phivals[key] = TraceError(msg)

        return phivals

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return self.facet_element.value_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented for the trace element.")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented for the trace element.")


def construct_dg_element(ref_el, degree):
    """Constructs a discontinuous galerkin element of a given degree
    on a particular reference cell.
    """
    if isinstance(ref_el, Simplex):
        dg_element = DiscontinuousLagrange(ref_el, degree)

    # Quadrilateral facets could be on a FiredrakeQuadrilateral.
    # In this case, we treat this as an interval x interval cell:
    elif isinstance(ref_el, FiredrakeQuadrilateral):
        dg_a = DiscontinuousLagrange(ufc_simplex(1), degree)
        dg_b = DiscontinuousLagrange(ufc_simplex(1), degree)
        dg_element = TensorProductElement(dg_a, dg_b)

    # This handles the more general case for facets:
    elif isinstance(ref_el, TensorProductCell):
        A, B = ref_el.cells

        if isinstance(A, Point) and not isinstance(B, Point):
            dg_element = construct_dg_element(B, degree)

        elif isinstance(B, Point) and not isinstance(A, Point):
            dg_element = construct_dg_element(A, degree)

        else:
            dg_a = construct_dg_element(A, degree)
            dg_b = construct_dg_element(B, degree)
            dg_element = TensorProductElement(dg_a, dg_b)
    else:
        raise NotImplementedError(
            "Reference cells of type %s not currently supported" % type(ref_el)
        )

    return dg_element


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
