# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.reference_element import TRIANGLE


class MorleyDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points."""

    def __init__(self, ref_el):
        entity_ids = {}
        nodes = []
        cur = 0

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("Morley only defined on triangles")

        # vertex point evaluations

        entity_ids[0] = {}
        for v in sorted(top[0]):
            nodes.append(functional.PointEvaluation(ref_el, verts[v]))

            entity_ids[0][v] = [cur]
            cur += 1

        # edge dof -- normal at each edge midpoint
        entity_ids[1] = {}
        for e in sorted(top[1]):
            pt = ref_el.make_points(1, e, 2)[0]
            n = functional.PointNormalDerivative(ref_el, e, pt)
            nodes.append(n)
            entity_ids[1][e] = [cur]
            cur += 1

        entity_ids[2] = {0: []}

        super().__init__(nodes, ref_el, entity_ids)


class Morley(finite_element.CiarletElement):
    """The Morley finite element."""

    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 2)
        dual = MorleyDualSet(ref_el)
        super().__init__(poly_set, dual, 2)
