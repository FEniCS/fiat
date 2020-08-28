# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified 2017 by RCK
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional


class CubicHermiteDualSet(dual_set.DualSet):
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
        sd = ref_el.get_spatial_dimension()

        # get jet at each vertex

        entity_ids[0] = {}
        for v in sorted(top[0]):
            nodes.append(functional.PointEvaluation(ref_el, verts[v]))
            pd = functional.PointDerivative
            for i in range(sd):
                alpha = [0] * sd
                alpha[i] = 1

                nodes.append(pd(ref_el, verts[v], alpha))

            entity_ids[0][v] = list(range(cur, cur + 1 + sd))
            cur += sd + 1

        # now only have dofs at the barycenter, which is the
        # maximal dimension
        # no edge dof

        entity_ids[1] = {}
        for i in top[1]:
            entity_ids
            entity_ids[1][i] = []

        if sd > 1:
            # face dof
            # point evaluation at barycenter
            entity_ids[2] = {}
            for f in sorted(top[2]):
                pt = ref_el.make_points(2, f, 3)[0]
                n = functional.PointEvaluation(ref_el, pt)
                nodes.append(n)
                entity_ids[2][f] = list(range(cur, cur + 1))
                cur += 1

            for dim in range(3, sd + 1):
                entity_ids[dim] = {}
                for facet in top[dim]:
                    entity_ids[dim][facet] = []

        super(CubicHermiteDualSet, self).__init__(nodes, ref_el, entity_ids)


class CubicHermite(finite_element.CiarletElement):
    """The cubic Hermite finite element.  It is what it is."""

    def __init__(self, ref_el, deg=3):
        assert deg == 3
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 3)
        dual = CubicHermiteDualSet(ref_el)

        super(CubicHermite, self).__init__(poly_set, dual, 3)
