# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.reference_element import TRIANGLE


class ArgyrisDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []
        cur = 0

        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()

        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("Argyris only defined on triangles")

        pe = functional.PointEvaluation
        pd = functional.PointDerivative
        pnd = functional.PointNormalDerivative

        # get jet at each vertex

        entity_ids[0] = {}
        for v in sorted(top[0]):
            nodes.append(pe(ref_el, verts[v]))

            # first derivatives
            for i in range(sd):
                alpha = [0] * sd
                alpha[i] = 1
                nodes.append(pd(ref_el, verts[v], alpha))

            # second derivatives
            alphas = [[2, 0], [1, 1], [0, 2]]
            for alpha in alphas:
                nodes.append(pd(ref_el, verts[v], alpha))

            entity_ids[0][v] = list(range(cur, cur + 6))
            cur += 6

        # edge dof
        entity_ids[1] = {}
        for e in sorted(top[1]):
            # normal derivatives at degree - 4 points on each edge
            ndpts = ref_el.make_points(1, e, degree - 3)
            ndnds = [pnd(ref_el, e, pt) for pt in ndpts]
            nodes.extend(ndnds)
            entity_ids[1][e] = list(range(cur, cur + len(ndpts)))
            cur += len(ndpts)

            # point value at degree-5 points on each edge
            if degree > 5:
                ptvalpts = ref_el.make_points(1, e, degree - 4)
                ptvalnds = [pe(ref_el, pt) for pt in ptvalpts]
                nodes.extend(ptvalnds)
                entity_ids[1][e] += list(range(cur, cur + len(ptvalpts)))
                cur += len(ptvalpts)

        # internal dof
        entity_ids[2] = {}
        if degree > 5:
            internalpts = ref_el.make_points(2, 0, degree - 3)
            internalnds = [pe(ref_el, pt) for pt in internalpts]
            nodes.extend(internalnds)
            entity_ids[2][0] = list(range(cur, cur + len(internalpts)))
            cur += len(internalpts)
        else:
            entity_ids[2] = {0: []}

        super(ArgyrisDualSet, self).__init__(nodes, ref_el, entity_ids)


class QuinticArgyrisDualSet(dual_set.DualSet):
    def __init__(self, ref_el):
        entity_ids = {}
        nodes = []
        cur = 0

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("Argyris only defined on triangles")

        pd = functional.PointDerivative

        # get jet at each vertex

        entity_ids[0] = {}
        for v in sorted(top[0]):
            nodes.append(functional.PointEvaluation(ref_el, verts[v]))

            # first derivatives
            for i in range(sd):
                alpha = [0] * sd
                alpha[i] = 1
                nodes.append(pd(ref_el, verts[v], alpha))

            # second derivatives
            alphas = [[2, 0], [1, 1], [0, 2]]
            for alpha in alphas:
                nodes.append(pd(ref_el, verts[v], alpha))

            entity_ids[0][v] = list(range(cur, cur + 6))
            cur += 6

        # edge dof -- normal at each edge midpoint
        entity_ids[1] = {}
        for e in sorted(top[1]):
            pt = ref_el.make_points(1, e, 2)[0]
            n = functional.PointNormalDerivative(ref_el, e, pt)
            nodes.append(n)
            entity_ids[1][e] = [cur]
            cur += 1

        entity_ids[2] = {0: []}

        super(QuinticArgyrisDualSet, self).__init__(nodes, ref_el, entity_ids)


class Argyris(finite_element.CiarletElement):
    """The Argyris finite element."""

    def __init__(self, ref_el, degree):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = ArgyrisDualSet(ref_el, degree)
        super(Argyris, self).__init__(poly_set, dual, degree)


class QuinticArgyris(finite_element.CiarletElement):
    """The Argyris finite element."""

    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 5)
        dual = QuinticArgyrisDualSet(ref_el)
        super().__init__(poly_set, dual, 5)
