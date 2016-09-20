# -*- coding: utf-8 -*-
"""Implementation of the Hellan-Herrmann-Johnson finite elements."""

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

from __future__ import absolute_import, print_function, division

from FIAT.finite_element import CiarletElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import ONSymTensorPolynomialSet
from FIAT.functional import PointwiseInnerProductEvaluation as InnerProduct
import numpy


class HellanHerrmannJohnsonDual(DualSet):
    """Degrees of freedom for Hellan-Herrmann-Johnson elements."""
    def __init__(self, cell, degree):
        dim = cell.get_spatial_dimension()
        if not dim == 2:
            raise ValueError("Hellan_Herrmann-Johnson elements are only"
                             "defined in dimension 2.")

        # construct the degrees of freedoms
        dofs = []               # list of functionals
        # dof_ids[i][j] contains the indices of dofs that are associated with
        # entity j in dim i
        dof_ids = {}

        # no vertex dof
        dof_ids[0] = {i: [] for i in range(dim + 1)}
        # edge dofs
        (_dofs, _dof_ids) = self._generate_edge_dofs(cell, degree, 0)
        dofs.extend(_dofs)
        dof_ids[1] = _dof_ids
        # cell dofs
        (_dofs, _dof_ids) = self._generate_trig_dofs(cell, degree, len(dofs))
        dofs.extend(_dofs)
        dof_ids[dim] = _dof_ids

        super(HellanHerrmannJohnsonDual, self).__init__(dofs, cell, dof_ids)

    @staticmethod
    def _generate_edge_dofs(cell, degree, offset):
        """generate dofs on edges.
        On each edge, let n be its normal. For degree=r, the scalar function
              n^T u n
        is evaluated at points enough to control P(r).
        """
        dofs = []
        dof_ids = {}
        for entity_id in range(3):                  # a triangle has 3 edges
            pts = cell.make_points(1, entity_id, degree + 2)  # edges are 1D
            normal = cell.compute_scaled_normal(entity_id)
            dofs += [InnerProduct(cell, normal, normal, pt) for pt in pts]
            num_new_dofs = len(pts)                 # 1 dof per point on edge
            dof_ids[entity_id] = list(range(offset, offset + num_new_dofs))
            offset += num_new_dofs
        return (dofs, dof_ids)

    @staticmethod
    def _generate_trig_dofs(cell, degree, offset):
        """generate dofs on edges.
        On each triangle, for degree=r, the three components
              u11, u12, u22
        are evaluated at points enough to control P(r-1).
        """
        dofs = []
        dof_ids = {}
        pts = cell.make_points(2, 0, degree + 2)  # 2D trig #0
        e1 = numpy.array([1.0, 0.0])              # euclidean basis 1
        e2 = numpy.array([0.0, 1.0])              # euclidean basis 2
        basis = [(e1, e1), (e1, e2), (e2, e2)]    # basis for symmetric matrix
        for (v1, v2) in basis:
            dofs += [InnerProduct(cell, v1, v2, pt) for pt in pts]
        num_dofs = 3 * len(pts)                   # 3 dofs per trig
        dof_ids[0] = list(range(offset, offset + num_dofs))
        return (dofs, dof_ids)


class HellanHerrmannJohnson(CiarletElement):
    """The definition of Hellan-Herrmann-Johnson element. It is defined only in
       dimension 2. It consists of piecewise polynomial symmetric-matrix-valued
       functions of degree r or less with normal-normal continuity.
    """
    def __init__(self, cell, degree):
        assert degree >= 0, "Hellan-Herrmann-Johnson starts at degree 0!"
        # shape functions
        Ps = ONSymTensorPolynomialSet(cell, degree)
        # degrees of freedom
        Ls = HellanHerrmannJohnsonDual(cell, degree)
        # mapping under affine transformation
        mapping = "double contravariant piola"

        super(HellanHerrmannJohnson, self).__init__(Ps, Ls, degree,
                                                    mapping=mapping)
