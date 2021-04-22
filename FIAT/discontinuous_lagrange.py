# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
from FIAT import finite_element, polynomial_set, dual_set, functional, P0
from FIAT.finite_element import make_barycentric_lattice_coordinates, make_index_permutations


def make_entity_permutations(dim, npoints):
    if npoints <= 0:
        return {o: [] for o in range(np.math.factorial(dim + 1))}
    a = make_barycentric_lattice_coordinates(dim, npoints - 1)
    index_perms = make_index_permutations(dim + 1)
    # Make DG nodes CG nodes map
    #
    # DG nodes are ordered by:
    # - group 0: entity dim
    # - group 1: entity ids
    # - cg counterpart node numbers (lexicographic)
    #
    # Ex: dim = 2, degree = 3
    #
    #     facet ids    cg node numbers
    #    +
    #    | \              3
    #    |   \  0         2 6
    #  2 |     \          1 5 8
    #    |       \        0 4 7 9
    #    +--------+
    #        1
    #
    # group0    = [ 0, 1, 1, 0, 1, 2, 1, 1, 1, 0]
    # group1    = [-3, 2, 2,-2, 1, 0, 0, 1, 0,-1]
    # cg        = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #
    # dg_cg_map = [ 0, 3, 9, 6, 8, 4, 7, 1, 2, 5]
    #
    g0 = dim - (a == 0).astype(int).sum(axis=1)
    g1 = np.zeros_like(g0)
    for d in range(dim + 1):
        on_facet_d = (a[:, d] == 0).astype(int)
        g1 += d * on_facet_d
    # Vertices are numbered differently from facets/edges
    # in FIAT, so we need to reverse.
    g1[g0 == 0] = -g1[g0 == 0]
    g0 = g0.reshape((a.shape[0], 1))
    g1 = g1.reshape((a.shape[0], 1))
    dg_cg_map = np.lexsort(np.transpose(np.concatenate((a, g1, g0), axis=1)))
    cg_dg_map = np.empty_like(dg_cg_map)
    for i, im in enumerate(dg_cg_map):
        cg_dg_map[im] = i
    perms = {}
    for o, index_perm in enumerate(index_perms):
        perm = np.lexsort(np.transpose(a[:, index_perm]))
        perm = cg_dg_map[perm][dg_cg_map]
        perms[o] = perm.tolist()
    return perms


class DiscontinuousLagrangeDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points.  This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""

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
            perms = make_entity_permutations(dim, degree + 1 if dim == len(top) - 1 else -1)
            for entity in sorted(top[dim]):
                pts_cur = ref_el.make_points(dim, entity, degree)
                nodes_cur = [functional.PointEvaluation(ref_el, x)
                             for x in pts_cur]
                nnodes_cur = len(nodes_cur)
                nodes += nodes_cur
                entity_ids[dim][entity] = []
                cur += nnodes_cur
                entity_permutations[dim][entity] = perms
        entity_ids[dim][0] = list(range(len(nodes)))

        super(DiscontinuousLagrangeDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class HigherOrderDiscontinuousLagrange(finite_element.CiarletElement):
    """The discontinuous Lagrange finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = DiscontinuousLagrangeDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(HigherOrderDiscontinuousLagrange, self).__init__(poly_set, dual, degree, formdegree)


def DiscontinuousLagrange(ref_el, degree):
    if degree == 0:
        return P0.P0(ref_el)
    else:
        return HigherOrderDiscontinuousLagrange(ref_el, degree)
