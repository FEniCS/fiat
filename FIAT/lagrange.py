# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.finite_element import make_barycentric_lattice_coordinates, make_index_permutations
import numpy as np


def make_entity_permutations(dim, npoints):
    r"""Make orientation-permutation map for the given
    simplex dimension, dim, and the number of points along
    each axis

    As an example, we first compute the orientation of a
    triangular cell:

       +                    +
       | \                  | \
       2   0               47   42
       |     \              |     \
       +--1---+             +--43--+
    FIAT canonical     Mapped example physical cell

    Suppose that the cone of the physical cell is given by:

    C = [47, 42, 43]

    FIAT facet to Physical facet map is given by:

    M = [42, 43, 47]

    Then the orientation of the cell is computed as:

    C.index(M[0]) = 1; C.remove(M[0])
    C.index(M[1]) = 1; C.remove(M[1])
    C.index(M[2]) = 0; C.remove(M[2])

    o = (1 * 2!) + (1 * 1!) + (0 * 0!) = 3

    For npoints = 3, there are 6 DoFs:

        2                   5
        1 4                 4 3
        0 3 5               2 1 0
    FIAT canonical     Physical cell canonical

    The permutation associated with o = 3 then is:

    [2, 4, 5, 1, 3, 0]

    The output of this function contains one such permutation
    for each orientation for the given simplex dimension and
    the number of points along each axis.
    """
    if npoints <= 0:
        return {o: [] for o in range(np.math.factorial(dim + 1))}
    a = make_barycentric_lattice_coordinates(dim, npoints - 1)
    index_perms = make_index_permutations(dim + 1)
    perms = {}
    for o, index_perm in enumerate(index_perms):
        perm = np.lexsort(np.transpose(a[:, index_perm]))
        perms[o] = perm.tolist()
    return perms


class LagrangeDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points."""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []
        entity_permutations = {}

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()

        cur = 0
        for dim in sorted(top):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            perms = {0: [0, ]} if dim == 0 else make_entity_permutations(dim, degree - dim)
            for entity in sorted(top[dim]):
                pts_cur = ref_el.make_points(dim, entity, degree)
                nodes_cur = [functional.PointEvaluation(ref_el, x)
                             for x in pts_cur]
                nnodes_cur = len(nodes_cur)
                nodes += nodes_cur
                entity_ids[dim][entity] = list(range(cur, cur + nnodes_cur))
                cur += nnodes_cur
                entity_permutations[dim][entity] = perms

        super(LagrangeDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class Lagrange(finite_element.CiarletElement):
    """The Lagrange finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = LagrangeDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        super(Lagrange, self).__init__(poly_set, dual, degree, formdegree)
