# -*- coding: utf-8 -*-
"""Implementation of the Mardal-Tai-Winther finite elements."""

# Copyright (C) 2019 Robert Kirby <robert.c.kirby@gmail.com>
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
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.functional import IntegralMoment, Functional, IntegralMomentOfDivergence

from FIAT.quadrature import GaussLegendreQuadratureLineRule, QuadratureRule, make_quadrature
from FIAT.reference_element import UFCInterval as interval
import numpy


class IntegralLegendreDirectionalMoment(IntegralMoment):
    """Momement of v.s against a Legendre polynomial over an edge"""
    def __init__(self, cell, s, entity, mom_deg, comp_deg):
        sd = cell.get_spatial_dimension()
        assert sd == 2
        shp = (sd,)
        quadpoints = comp_deg + 1
        Q = GaussLegendreQuadratureLineRule(interval(), quadpoints)
        legendre = numpy.polynomial.legendre.legval(2*Q.get_points()-1, [0]*mom_deg + [1]) * cell.volume_of_subcomplex(1, entity)
        f_at_qpts = numpy.array([s*legendre[i] for i in range(quadpoints)])
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
            result[i, :] = numpy.dot(bfs, wts[:, i])

        return result
        

class IntegralLegendreNormalMoment(IntegralLegendreDirectionalMoment):
    """Momement of v.n against a Legendre polynomial over an edge"""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_normal(entity)
        IntegralLegendreDirectionalMoment.__init__(self, cell, n, entity,
                                                   mom_deg, comp_deg)
        

class IntegralLegendreTangentialMoment(IntegralLegendreDirectionalMoment):
    """Momement of v.t against a Legendre polynomial over an edge"""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        t = cell.compute_normalized_edge_tangent(entity)
        IntegralLegendreDirectionalMoment.__init__(self, cell, t, entity,
                                                   mom_deg, comp_deg)


def DivergenceDubinerMoments(cell, start_deg, stop_deg, comp_deg):
    onp = ONPolynomialSet(cell, stop_deg)
    Q = make_quadrature(cell, comp_deg)

    pts, wts = Q.get_points(), Q.get_weights()
    onp = onp.tabulate(pts, 0)[0,0]

    ells = []
        
    for ii in range((start_deg)*(start_deg+1)//2,
                    (stop_deg+1)*(stop_deg+2)//2):
        ells.append(IntegralMomentOfDivergence(cell, Q, onp[ii,:]))

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
        """generate dofs on edges.
        On each edge, let n be its normal.  We need to integrate
        u.n and u.t against the first Legendre polynomial (constant) 
        and u.n against the second (linear).
        """
        dofs = []
        dof_ids = {}
        offset = 0

        for entity_id in range(3):     # a triangle has 3 edges
            dofs += [IntegralLegendreNormalMoment(cell, entity_id, 0, 6),
                     IntegralLegendreTangentialMoment(cell, entity_id, 0, 6),
                     IntegralLegendreNormalMoment(cell, entity_id, 1, 6)]

            num_new_dofs = 3
            dof_ids[entity_id] = list(range(offset, offset + num_new_dofs))
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
        assert cell.get_spatial_dimension() == 2,  "Only defined for dimension 2"
        # polynomial space
        Ps = ONPolynomialSet(cell, degree, (2,))

        # degrees of freedom
        Ls = MardalTaiWintherDual(cell, degree)

        # mapping under affine transformation
        mapping = "contravariant piola"

        super(MardalTaiWinther, self).__init__(Ps, Ls, degree,
                                               mapping=mapping)
