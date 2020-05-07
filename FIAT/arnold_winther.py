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
from FIAT.functional import PointwiseInnerProductEvaluation as InnerProduct, IntegralMoment, FrobeniusIntegralMoment as FIM
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

class ArnoldWintherBaseDual(DualSet):
    """Parent class for the dofs of the conforming and 
    nonconforming Arnold-Winther elements. """

    def __init__(self, cell, degree):
        dim = cell.get_spatial_dimension()
        if not dim == 2:
            raise ValueError("Arnold-Winther elements are only"
                             "defined in dimension 2, for now! The theory"
                             "is there in 3D, I just haven't implemented it.")

        # construct the degrees of freedom
        dofs = []               # list of functionals
        # dof_ids[i][j] contains the indices of dofs that are associated with
        # entity j in dim i
        dof_ids = {}

        # no vertex dofs for nonconforming, but is called by conforming
        (_dofs, _dof_ids) = self._generate_vertex_dofs(cell, degree, len(dofs))
        dofs.extend(_dofs)
        dof_ids[0] = _dof_ids

        # edge dofs
        (_dofs, _dof_ids) = self._generate_edge_dofs(cell, degree)
        dofs.extend(_dofs)
        dof_ids[1] = _dof_ids

        # cell dofs
        (_dofs, _dof_ids) = self._generate_internal_dofs(cell, degree, len(dofs))
        dofs.extend(_dofs)
        dof_ids[dim] = _dof_ids

        # extra dofs for enforcing linearity of dot(n, dot(sigma, n)) on edges
        # (nonconforming case), or linearity of div (conforming case)
        (_dofs, _dof_ids) = self._generate_constraint_dofs(cell, degree, len(dofs))
        dofs.extend(_dofs)

        for entity_id in range(3):
            dof_ids[1][entity_id] = dof_ids[1][entity_id] + _dof_ids[entity_id]

        super(ArnoldWintherBaseDual, self).__init__(dofs, cell, dof_ids)

    @staticmethod
    def _generate_edge_dofs(cell, degree):
        """generate dofs on edges.
        On each edge, let n be its normal. The components of dot(u, n)
        are evaluated at two different points.
        """
        dofs = []
        dof_ids = {}
        offset = 0

        for entity_id in range(3):                  # a triangle has 3 edges
            for order in (0,1):
                dofs += [IntegralNormalNormalLegendreMoment(cell, entity_id, order, 3),
                         IntegralNormalTangentialLegendreMoment(cell, entity_id, order, 3)]
            num_new_dofs = 4                # 2 dof per order on edge
            dof_ids[entity_id] = list(range(offset, offset + num_new_dofs))
            offset += num_new_dofs
        return (dofs, dof_ids)


    @staticmethod
    def _generate_internal_dofs(cell, degree, offset):
        """generate internal dofs on the cell.
        On each triangle, for degree = r, the three components
              u11, u12, u22
        are evaluated at a single point.
        """
        dofs = []
        dof_ids = {}
        momdeg = degree - 2
        # Tabulate all ON polys of certain degree
        Q = make_quadrature(cell, 2*(degree+momdeg))
        sd = cell.get_spatial_dimension()
        Pkm2 = ONPolynomialSet(cell, momdeg)
        pkm2vals = Pkm2.tabulate(Q.get_points())[tuple([0 for i in range(sd)])]

        e1 = numpy.array([1.0, 0.0])              # euclidean basis 1
        e2 = numpy.array([0.0, 1.0])              # euclidean basis 2
        basis = [(e1, e1), (e1, e2), (e2, e2)]    # basis for symmetric matrices
        for (v1, v2) in basis:
            v1v2t = numpy.outer(v1, v2)
            for i in range(pkm2vals.shape[0]):
                fatqp = numpy.asarray([[[x * p for p in pkm2vals[i, :]]
                                        for x in y] for y in v1v2t])
                dofs.append(FIM(cell, Q, fatqp))

        num_dofs = len(dofs)
        dof_ids[0] = list(range(offset, offset + num_dofs))
        return (dofs, dof_ids)


class ArnoldWintherNCDual(ArnoldWintherBaseDual):
    """Degrees of freedom for nonconforming Arnold-Winther elements."""
    
    def __init__(self, cell, degree):
        if not degree == 2:
            raise ValueError("Nonconforming Arnold-Winther elements are" 
                             "only defined for degree 2.")

        ArnoldWintherBaseDual.__init__(self, cell, degree)

    @staticmethod
    def _generate_constraint_dofs(cell, degree, offset):
        """
        Generate constraint dofs on edges.
        dot(n, dot(sigma, n)) must be linear on each edge.
        So we introduce functionals whose kernel describes this property,
        as described in the FIAT paper.
        """
        dofs = []
        dof_ids = {}

        for entity_id in range(3):
            dof = IntegralNormalNormalLegendreMoment(cell, entity_id, degree, 2)
            dofs += [dof]
            dof_ids[entity_id] = [offset]
            offset += 1

        return (dofs, dof_ids)


    @staticmethod
    def _generate_vertex_dofs(cell, degree, offset):
        """No vertex dofs for the nonconforming element."""
        _dofs = []
        dim = cell.get_spatial_dimension()
        _dof_ids = {i: [] for i in range(dim + 1)}
        return (_dofs, _dof_ids)


class ArnoldWintherNC(CiarletElement):
    """The definition of the nonconforming Arnold-Winther element.
    """
    def __init__(self, cell, degree):
        assert degree == 2, "Only defined for degree 2"
        # polynomial space
        Ps = ONSymTensorPolynomialSet(cell, degree)
        # degrees of freedom
        Ls = ArnoldWintherNCDual(cell, degree)
        # mapping under affine transformation
        mapping = "double contravariant piola"

        super(ArnoldWintherNC, self).__init__(Ps, Ls, degree,
                                                  mapping=mapping)

class ArnoldWintherDual(ArnoldWintherBaseDual):
    """Degrees of freedom for conforming Arnold-Winther elements."""

    def __init__(self, cell, degree):
        if not degree == 3:
            raise ValueError("Conforming Arnold-Winther elements"
                             "are only defined for degree 3.")

        ArnoldWintherBaseDual.__init__(self, cell, degree)

    @staticmethod
    def _generate_internal_dofs(cell, degree, offset):
        """ This is as for the nonconforming element, except the 
        degree of the original polynomial space is 1 higher
        here. """
        return ArnoldWintherBaseDual._generate_internal_dofs(cell, degree-1, offset)

    @staticmethod
    def _generate_vertex_dofs(cell, degree, offset):
        """generate dofs of evaluation at vertices.

        """
        dofs = []
        dof_ids = {}
        offset = 0

        vs = numpy.array(cell.get_vertices())
        e1 = numpy.array([1.0, 0.0])
        e2 = numpy.array([0.0, 1.0])
        basis = [(e1, e1), (e1, e2), (e2, e2)]

        for entity_id in range(3):
            node = vs[entity_id]
            for (v1, v2) in basis:
                dofs.append(InnerProduct(cell, v1, v2, node))

            num_new_dofs = 3                # 3 components to evaluate per vertex
            dof_ids[entity_id] = list(range(offset, offset + num_new_dofs))
            offset += num_new_dofs

        return (dofs, dof_ids)

    @staticmethod
    def _generate_constraint_dofs(cell, degree, offset):
        """Generate constraint dofs. div(sigma) must be
        of degree "degree - 2", so we introduce functionals 
        whose kernel describes this property."""
        dofs = []
        dof_ids = {}

        dim = cell.get_spatial_dimension()
        
        # [DELETE] ONPs need to have degree 1 more than
        #          what is required of div.
        # or ONSymTensorPolynomialSet?
        onp = ONPolynomialSet(cell, degree-1)
        pts = Q.get_points()
        # [DELETE] should quad pts be tabulated into a
        #          tensor?
        #onp = onp.tabulate(pts)[tuple([[0 for j in range(dim)] for i in range(dim)])]
        onp = onp.tabulate(pts)[0, 0]

        # [DELETE] 2nd argument is how many quad points are required per spatial
        #          dimension. If N is the degree of the integrand, this is
        #          ceil( (N + 1)/2 )
        #          Here, N = 2*(degree - 1). So for general "degree", 2nd arg
        #          here should be    numpy.ceil(degree-0.5)

        #Q = make_quadrature(cell, numpy.ceil(degree-0.5))
        Q = make_quadrature(cell, 3)
        
        #for i in range(dim):
        #    """Extract the rows of the divergence."""
        #    e_i = numpy.zeros((1, dim), dtype=object)
        #    e_i[i+1] = 1.0
        #    row_i_at_qpts = numpy.asarray([[x * p for p in onp[i, :]]
        #                                    for x in e_i]) 
        #    dof = IntegralMomentOfDivergence(cell, Q, row_i_at_qpts)
        #    dofs += dof
        #    dof_ids[i] = [offset]
        #    offset += 1

        # [DELETE] Could use DivergenceDubinerMoments from mardal_tai_winther
        #          here, which the loop below mimics? Or, simply put that
        #          into functional.py

        # Take divergence against all the ONPs of degree exactly 
        # "degree-1".
        for i in range((degree-2)*(degree-1)//2, 
                           (degree-1)*degree//2):
            dofs.append(IntegralMomentOfDivergence(cell, Q, onp[i, :]))

        DOF_ids = list(range(offset, offset+len(dofs)))
        dof_ids.extend(DOF_ids)

        return (dofs, dof_ids)
        #return ([], None)

class ArnoldWinther(CiarletElement):
    """The definition of the conforming Arnold-Winther element.
    """
    def __init__(self, cell, degree):
        assert degree == 3, "Only defined for degree 3"
        # polynomial space
        Ps = ONSymTensorPolynomialSet(cell, degree)
        # degrees of freedom
        Ls = ArnoldWintherDual(cell, degree)
        # mapping under affine transformation
        mapping = "double contravariant piola"

        super(ArnoldWinther, self).__init__(Ps, Ls, degree,
                                                mapping=mapping)
