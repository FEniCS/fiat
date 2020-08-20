# -*- coding: utf-8 -*-
"""Implementation of the Mardal-Tai-Winther finite elements."""

# Copyright (C) 2020 by Robert C. Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from FIAT.finite_element import CiarletElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.functional import (IntegralMomentOfNormalEvaluation,
                             IntegralMomentOfTangentialEvaluation,
                             IntegralLegendreNormalMoment,
                             IntegralMomentOfDivergence)

from FIAT.quadrature import make_quadrature


def DivergenceDubinerMoments(cell, start_deg, stop_deg, comp_deg):
    onp = ONPolynomialSet(cell, stop_deg)
    Q = make_quadrature(cell, comp_deg)

    pts = Q.get_points()
    onp = onp.tabulate(pts, 0)[0, 0]

    ells = []

    for ii in range((start_deg)*(start_deg+1)//2,
                    (stop_deg+1)*(stop_deg+2)//2):
        ells.append(IntegralMomentOfDivergence(cell, Q, onp[ii, :]))

    return ells


class MardalTaiWintherDual(DualSet):
    """Degrees of freedom for Mardal-Tai-Winther elements."""
    def __init__(self, cell, degree):
        dim = cell.get_spatial_dimension()
        if not dim == 2:
            raise ValueError("Mardal-Tai-Winther elements are only"
                             "defined in dimension 2.")

        if not degree == 3:
            raise ValueError("Mardal-Tai-Winther elements are only defined"
                             "for degree 3.")

        # construct the degrees of freedoms
        dofs = []               # list of functionals

        # dof_ids[i][j] contains the indices of dofs that are associated with
        # entity j in dim i
        dof_ids = {}

        # no vertex dof
        dof_ids[0] = {i: [] for i in range(dim + 1)}

        # edge dofs
        (_dofs, _dof_ids) = self._generate_edge_dofs(cell, degree)
        dofs.extend(_dofs)
        dof_ids[1] = _dof_ids

        # no cell dofs
        dof_ids[2] = {}
        dof_ids[2][0] = []

        # extra dofs for enforcing div(v) constant over the cell and
        # v.n linear on edges
        (_dofs, _edge_dof_ids, _cell_dof_ids) = self._generate_constraint_dofs(cell, degree, len(dofs))
        dofs.extend(_dofs)

        for entity_id in range(3):
            dof_ids[1][entity_id] = dof_ids[1][entity_id] + _edge_dof_ids[entity_id]

        dof_ids[2][0] = dof_ids[2][0] + _cell_dof_ids

        super(MardalTaiWintherDual, self).__init__(dofs, cell, dof_ids)

    @staticmethod
    def _generate_edge_dofs(cell, degree):
        """Generate dofs on edges.
        On each edge, let n be its normal.  We need to integrate
        u.n and u.t against the first Legendre polynomial (constant)
        and u.n against the second (linear).
        """
        dofs = []
        dof_ids = {}
        offset = 0
        sd = 2

        facet = cell.get_facet_element()
        # Facet nodes are \int_F v\cdot n p ds where p \in P_{q-1}
        # degree is q - 1
        Q = make_quadrature(facet, 6)
        Pq = ONPolynomialSet(facet, 1)
        Pq_at_qpts = Pq.tabulate(Q.get_points())[tuple([0]*(sd - 1))]
        for f in range(3):
            phi0 = Pq_at_qpts[0, :]
            dofs.append(IntegralMomentOfNormalEvaluation(cell, Q, phi0, f))
            dofs.append(IntegralMomentOfTangentialEvaluation(cell, Q, phi0, f))
            phi1 = Pq_at_qpts[1, :]
            dofs.append(IntegralMomentOfNormalEvaluation(cell, Q, phi1, f))

            num_new_dofs = 3
            dof_ids[f] = list(range(offset, offset + num_new_dofs))
            offset += num_new_dofs

        return (dofs, dof_ids)

    @staticmethod
    def _generate_constraint_dofs(cell, degree, offset):
        """
        Generate constraint dofs on the cell and edges
        * div(v) must be constant on the cell.  Since v is a cubic and
          div(v) is quadratic, we need the integral of div(v) against the
          linear and quadratic Dubiner polynomials to vanish.
          There are two linear and three quadratics, so these are five
          constraints
        * v.n must be linear on each edge.  Since v.n is cubic, we need
          the integral of v.n against the cubic and quadratic Legendre
          polynomial to vanish on each edge.

        So we introduce functionals whose kernel describes this property,
        as described in the FIAT paper.
        """
        dofs = []

        edge_dof_ids = {}
        for entity_id in range(3):
            dofs += [IntegralLegendreNormalMoment(cell, entity_id, 2, 6),
                     IntegralLegendreNormalMoment(cell, entity_id, 3, 6)]

            edge_dof_ids[entity_id] = [offset, offset+1]
            offset += 2

        cell_dofs = DivergenceDubinerMoments(cell, 1, 2, 6)
        dofs.extend(cell_dofs)
        cell_dof_ids = list(range(offset, offset+len(cell_dofs)))

        return (dofs, edge_dof_ids, cell_dof_ids)


class MardalTaiWinther(CiarletElement):
    """The definition of the Mardal-Tai-Winther element.
    """
    def __init__(self, cell, degree=3):
        assert degree == 3, "Only defined for degree 3"
        assert cell.get_spatial_dimension() == 2, "Only defined for dimension 2"
        # polynomial space
        Ps = ONPolynomialSet(cell, degree, (2,))

        # degrees of freedom
        Ls = MardalTaiWintherDual(cell, degree)

        # mapping under affine transformation
        mapping = "contravariant piola"

        super(MardalTaiWinther, self).__init__(Ps, Ls, degree,
                                               mapping=mapping)
