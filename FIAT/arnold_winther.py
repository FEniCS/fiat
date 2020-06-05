# -*- coding: utf-8 -*-
"""Implementation of the Arnold-Winther finite elements."""

# Copyright (C) 2016-2018 Lizao Li <lzlarryli@gmail.com>
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

from FIAT.finite_element import CiarletElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import ONSymTensorPolynomialSet, ONPolynomialSet
from FIAT.functional import PointwiseInnerProductEvaluation as InnerProduct, IntegralMoment, FrobeniusIntegralMoment as FIM, IntegralMomentOfTensorDivergence
from FIAT.quadrature import GaussLegendreQuadratureLineRule, QuadratureRule, make_quadrature
from FIAT.reference_element import UFCInterval as interval
import numpy


class IntegralBidirectionalLegendreMoment(IntegralMoment):
    """Moment of dot(s1, dot(tau, s2)) against Legendre on entity, multiplied by the size of the reference facet"""
    def __init__(self, cell, s1, s2, entity, mom_deg, comp_deg):
        # mom_deg is degree of moment, comp_deg is the total degree of
        # polynomial you might need to integrate (or something like that)
        sd = cell.get_spatial_dimension()
        shp = (sd, sd)

        s1s2T = numpy.outer(s1, s2)
        quadpoints = comp_deg + 1
        Q = GaussLegendreQuadratureLineRule(interval(), quadpoints)

        # The volume squared gets the Jacobian mapping from line interval
        # and the edge length into the functional.
        legendre = numpy.polynomial.legendre.legval(2*Q.get_points()-1, [0]*mom_deg + [1]) * numpy.abs(cell.volume_of_subcomplex(1, entity))**2

        f_at_qpts = numpy.array([s1s2T*legendre[i] for i in range(quadpoints)])

        # Map the quadrature points
        fmap = cell.get_entity_transform(sd-1, entity)
        mappedqpts = [fmap(pt) for pt in Q.get_points()]
        mappedQ = QuadratureRule(cell, mappedqpts, Q.get_weights())

        IntegralMoment.__init__(self, cell, mappedQ, f_at_qpts, shp=shp)

    def to_riesz(self, poly_set):
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        pts = list(self.pt_dict.keys())
        bfs = es.tabulate(ed, pts)
        wts = numpy.array([foo[0][0] for foo in list(self.pt_dict.values())])
        result = numpy.zeros(poly_set.coeffs.shape[1:], "d")

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j, :] = numpy.dot(bfs, wts[:, i, j])

        return result


class IntegralNormalNormalLegendreMoment(IntegralBidirectionalLegendreMoment):
    """Moment of dot(n, dot(tau, n)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_normal(entity)
        IntegralBidirectionalLegendreMoment.__init__(self, cell, n, n,
                                                     entity, mom_deg, comp_deg)


class IntegralNormalTangentialLegendreMoment(IntegralBidirectionalLegendreMoment):
    """Moment of dot(n, dot(tau, n)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_normal(entity)
        t = cell.compute_normalized_edge_tangent(entity)
        IntegralBidirectionalLegendreMoment.__init__(self, cell, n, t,
                                                     entity, mom_deg, comp_deg)


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
                dofs += [IntegralNormalNormalLegendreMoment(cell, entity_id, order, 6),
                         IntegralNormalTangentialLegendreMoment(cell, entity_id, order, 6)]
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
            dof = IntegralNormalNormalLegendreMoment(cell, entity_id, 2, 6)
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
                dofs += [IntegralNormalNormalLegendreMoment(cell, entity_id, order, 6),
                         IntegralNormalTangentialLegendreMoment(cell, entity_id, order, 6)]
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
