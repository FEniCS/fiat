# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Miklós Homolya
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with FIAT.  If not, see <https://www.gnu.org/licenses/>.

import itertools
import math

import numpy

from FIAT.finite_element import FiniteElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import mis


class BernsteinDualSet(DualSet):
    """The dual basis for Bernstein elements."""

    def __init__(self, ref_el, degree):
        # Initialise data structures
        topology = ref_el.get_topology()
        entity_ids = {dim: {entity_i: []
                            for entity_i in entities}
                      for dim, entities in topology.items()}

        # Calculate inverse topology
        inverse_topology = {vertices: (dim, entity_i)
                            for dim, entities in topology.items()
                            for entity_i, vertices in entities.items()}

        # Generate triangular barycentric indices
        dim = ref_el.get_spatial_dimension()
        alphas = [(degree - sum(beta),) + tuple(reversed(beta))
                  for beta in itertools.product(range(degree + 1), repeat=dim)
                  if sum(beta) <= degree]

        # Fill data structures
        nodes = []
        for i, alpha in enumerate(alphas):
            vertices, = numpy.nonzero(alpha)
            entity_dim, entity_i = inverse_topology[tuple(vertices)]
            entity_ids[entity_dim][entity_i].append(i)

            # Leave nodes unimplemented for now
            nodes.append(None)

        super(BernsteinDualSet, self).__init__(nodes, ref_el, entity_ids)


class Bernstein(FiniteElement):
    """A finite element with Bernstein polynomials as basis functions."""

    def __init__(self, ref_el, degree):
        dual = BernsteinDualSet(ref_el, degree)
        k = 0  # 0-form
        super(Bernstein, self).__init__(ref_el, dual, degree, k)

    def degree(self):
        """The degree of the polynomial space."""
        return self.get_order()

    def value_shape(self):
        """The value shape of the finite element functions."""
        return ()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        # Transform points to reference cell coordinates
        ref_el = self.get_reference_element()
        if entity is None:
            entity = (ref_el.get_spatial_dimension(), 0)

        entity_dim, entity_id = entity
        entity_transform = ref_el.get_entity_transform(entity_dim, entity_id)
        cell_points = list(map(entity_transform, points))

        # Construct Cartesian to Barycentric coordinate mapping
        vs = numpy.asarray(ref_el.get_vertices())
        B2R = numpy.vstack([vs.T, numpy.ones(len(vs))])
        R2B = numpy.linalg.inv(B2R)

        B = numpy.hstack([cell_points,
                          numpy.ones((len(cell_points), 1))]).dot(R2B.T)
        # X = sympy.symbols('X Y Z')[:dim]
        # B = R2B.dot(X + (1,))

        # Generate triangular barycentric indices
        deg = self.degree()
        dim = ref_el.get_spatial_dimension()
        etas = [(deg - sum(beta),) + tuple(reversed(beta))
                for beta in itertools.product(range(deg + 1), repeat=dim)
                if sum(beta) <= deg]

        result = {}
        for D in range(order + 1):
            for alpha in mis(dim, D):
                table = numpy.zeros((len(etas), len(cell_points)))
                for i, eta in enumerate(etas):
                    table[i, :] = bernstein_dx(B, eta, alpha, R2B)
                    # for j, point in enumerate(cell_points):
                    #     # c = math.factorial(deg)
                    #     # for k in eta:
                    #     #     c = c // math.factorial(k)
                    #     b = c * (B ** eta).prod()
                    #     e = sympy.diff(b, *zip(X, alpha))
                    #     table[i, j] = e.subs(dict(zip(X, point))).evalf()
                result[alpha] = table
        return result


def bernstein_b(points, alpha):
    """Evaluates Bernstein polynomials at barycentric points.

    :arg points: array of points in barycentric coordinates
    :arg alpha: exponents defining the Bernstein polynomial
    :returns: array of Bernstein polynomial values at given points.
    """
    points = numpy.asarray(points)
    alpha = tuple(alpha)

    N, d_1 = points.shape
    assert d_1 == len(alpha)
    if any(k < 0 for k in alpha):
        return numpy.zeros(len(points))
    elif all(k == 0 for k in alpha):
        return numpy.ones(len(points))
    else:
        c = math.factorial(sum(alpha))
        for k in alpha:
            c = c // math.factorial(k)
        return c * numpy.prod(points**alpha, axis=1)


def bernstein_db(points, alpha, delta):
    points = numpy.asarray(points)
    alpha = tuple(alpha)
    delta = tuple(delta)

    N, d_1 = points.shape
    assert d_1 == len(alpha) == len(delta)

    # Calculate derivative factor
    c = 1
    for _, i in zip(range(sum(delta)), range(sum(alpha), 0, -1)):
        c *= i

    alpha_ = numpy.array(alpha) - numpy.array(delta)
    return c * bernstein_b(points, alpha_)


def bernstein_Db(points, alpha, order):
    points = numpy.asarray(points)
    alpha = tuple(alpha)

    N, d_1 = points.shape
    assert d_1 == len(alpha)
    Dshape = (d_1,) * order

    result = numpy.empty(Dshape + (N,))
    for indices in numpy.ndindex(Dshape):
        delta = [0] * d_1
        for i in indices:
            delta[i] += 1
        result[indices + (slice(None),)] = bernstein_db(points, alpha, delta)
    return result


def bernstein_dx(points, alpha, delta, R2B):
    points = numpy.asarray(points)
    alpha = tuple(alpha)
    delta = tuple(delta)

    N, d_1 = points.shape
    assert d_1 == len(alpha) == len(delta) + 1

    result = bernstein_Db(points, alpha, sum(delta))
    for d, c in enumerate(delta):
        for _ in range(c):
            result = R2B[:, d].dot(result)
    return result
