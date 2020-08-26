# -*- coding: utf-8 -*-
"""Implementation of the Arnold-Winther finite elements."""

# Copyright (C) 2020 by Robert C. Kirby (Baylor University)
# Modified by Francis Aznaran (Oxford University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT.finite_element import CiarletElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import ONSymTensorPolynomialSet, ONPolynomialSet
from FIAT.functional import (
    PointwiseInnerProductEvaluation as InnerProduct,
    FrobeniusIntegralMoment as FIM,
    IntegralMomentOfTensorDivergence,
    IntegralLegendreNormalNormalMoment,
    IntegralLegendreNormalTangentialMoment)

from FIAT.quadrature import make_quadrature

import numpy


class ArnoldWintherNCDual(DualSet):
    def __init__(self, cell, degree):
        if not degree == 2:
            raise ValueError("Nonconforming Arnold-Winther elements are"
                             "only defined for degree 2.")
        dofs = []
        dof_ids = {}
        dof_ids[0] = {0: [], 1: [], 2: []}
        dof_ids[1] = {0: [], 1: [], 2: []}
        dof_ids[2] = {0: []}

        dof_cur = 0
        # no vertex dofs
        # proper edge dofs now (not the contraints)
        # moments of normal . sigma against constants and linears.
        for entity_id in range(3):                  # a triangle has 3 edges
            for order in (0, 1):
                dofs += [IntegralLegendreNormalNormalMoment(cell, entity_id, order, 6),
                         IntegralLegendreNormalTangentialMoment(cell, entity_id, order, 6)]
            dof_ids[1][entity_id] = list(range(dof_cur, dof_cur+4))
            dof_cur += 4

        # internal dofs: constant moments of three unique components
        Q = make_quadrature(cell, 2)

        e1 = numpy.array([1.0, 0.0])              # euclidean basis 1
        e2 = numpy.array([0.0, 1.0])              # euclidean basis 2
        basis = [(e1, e1), (e1, e2), (e2, e2)]    # basis for symmetric matrices
        for (v1, v2) in basis:
            v1v2t = numpy.outer(v1, v2)
            fatqp = numpy.zeros((2, 2, len(Q.pts)))
            for i, y in enumerate(v1v2t):
                for j, x in enumerate(y):
                    for k in range(len(Q.pts)):
                        fatqp[i, j, k] = x
            dofs.append(FIM(cell, Q, fatqp))
        dof_ids[2][0] = list(range(dof_cur, dof_cur + 3))
        dof_cur += 3

        # put the constraint dofs last.
        for entity_id in range(3):
            dof = IntegralLegendreNormalNormalMoment(cell, entity_id, 2, 6)
            dofs.append(dof)
            dof_ids[1][entity_id].append(dof_cur)
            dof_cur += 1

        super(ArnoldWintherNCDual, self).__init__(dofs, cell, dof_ids)


class ArnoldWintherNC(CiarletElement):
    """The definition of the nonconforming Arnold-Winther element.
    """
    def __init__(self, cell, degree):
        assert degree == 2, "Only defined for degree 2"
        Ps = ONSymTensorPolynomialSet(cell, degree)
        Ls = ArnoldWintherNCDual(cell, degree)
        mapping = "double contravariant piola"

        super(ArnoldWintherNC, self).__init__(Ps, Ls, degree,
                                              mapping=mapping)


class ArnoldWintherDual(DualSet):
    def __init__(self, cell, degree):
        if not degree == 3:
            raise ValueError("Arnold-Winther elements are"
                             "only defined for degree 3.")
        dofs = []
        dof_ids = {}
        dof_ids[0] = {0: [], 1: [], 2: []}
        dof_ids[1] = {0: [], 1: [], 2: []}
        dof_ids[2] = {0: []}

        dof_cur = 0

        # vertex dofs
        vs = cell.get_vertices()
        e1 = numpy.array([1.0, 0.0])
        e2 = numpy.array([0.0, 1.0])
        basis = [(e1, e1), (e1, e2), (e2, e2)]

        dof_cur = 0

        for entity_id in range(3):
            node = tuple(vs[entity_id])
            for (v1, v2) in basis:
                dofs.append(InnerProduct(cell, v1, v2, node))
            dof_ids[0][entity_id] = list(range(dof_cur, dof_cur + 3))
            dof_cur += 3

        # edge dofs now
        # moments of normal . sigma against constants and linears.
        for entity_id in range(3):
            for order in (0, 1):
                dofs += [IntegralLegendreNormalNormalMoment(cell, entity_id, order, 6),
                         IntegralLegendreNormalTangentialMoment(cell, entity_id, order, 6)]
            dof_ids[1][entity_id] = list(range(dof_cur, dof_cur+4))
            dof_cur += 4

        # internal dofs: constant moments of three unique components
        Q = make_quadrature(cell, 3)

        e1 = numpy.array([1.0, 0.0])              # euclidean basis 1
        e2 = numpy.array([0.0, 1.0])              # euclidean basis 2
        basis = [(e1, e1), (e1, e2), (e2, e2)]    # basis for symmetric matrices
        for (v1, v2) in basis:
            v1v2t = numpy.outer(v1, v2)
            fatqp = numpy.zeros((2, 2, len(Q.pts)))
            for k in range(len(Q.pts)):
                fatqp[:, :, k] = v1v2t
            dofs.append(FIM(cell, Q, fatqp))
        dof_ids[2][0] = list(range(dof_cur, dof_cur + 3))
        dof_cur += 3

        # Constraint dofs

        Q = make_quadrature(cell, 5)

        onp = ONPolynomialSet(cell, 2, (2,))
        pts = Q.get_points()
        onpvals = onp.tabulate(pts)[0, 0]

        for i in list(range(3, 6)) + list(range(9, 12)):
            dofs.append(IntegralMomentOfTensorDivergence(cell, Q,
                                                         onpvals[i, :, :]))

        dof_ids[2][0] += list(range(dof_cur, dof_cur+6))

        super(ArnoldWintherDual, self).__init__(dofs, cell, dof_ids)


class ArnoldWinther(CiarletElement):
    """The definition of the conforming Arnold-Winther element.
    """
    def __init__(self, cell, degree):
        assert degree == 3, "Only defined for degree 3"
        Ps = ONSymTensorPolynomialSet(cell, degree)
        Ls = ArnoldWintherDual(cell, degree)
        mapping = "double contravariant piola"
        super(ArnoldWinther, self).__init__(Ps, Ls, degree, mapping=mapping)
