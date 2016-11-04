# -*- coding: utf-8 -*-
"""Implementation of the generalized Regge finite elements."""

# Copyright (C) 2015-2018 Lizao Li
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


class ReggeDual(DualSet):
    """Degrees of freedom for generalized Regge finite elements."""
    def __init__(self, cell, degree):
        dim = cell.get_spatial_dimension()
        if (dim < 2) or (dim > 3):
            raise ValueError("Generalized Regge elements are implemented only "
                             "for dimension 2--3. For 1D, it is just DG(r).")

        # construct the degrees of freedoms
        dofs = []               # list of functionals
        # dof_ids[i][j] contains the indices of dofs that are associated with
        # entity j in dim i
        dof_ids = {}

        # no vertex dof
        dof_ids[0] = {i: [] for i in range(dim + 1)}
        # edge dofs
        (_dofs, _dof_ids) = self._generate_dofs(cell, 1, degree, 0)
        dofs.extend(_dofs)
        dof_ids[1] = _dof_ids
        # facet dofs for 3D
        if dim == 3:
            (_dofs, _dof_ids) = self._generate_dofs(cell, 2, degree, len(dofs))
            dofs.extend(_dofs)
            dof_ids[2] = _dof_ids
        # cell dofs
        (_dofs, _dof_ids) = self._generate_dofs(cell, dim, degree, len(dofs))
        dofs.extend(_dofs)
        dof_ids[dim] = _dof_ids

        super(ReggeDual, self).__init__(dofs, cell, dof_ids)

    @staticmethod
    def _generate_dofs(cell, entity_dim, degree, offset):
        """generate degrees of freedom for enetities of dimension entity_dim

        Input: all obvious except
           offset  -- the current first available dof id.

        Output:
           dofs    -- an array of dofs associated to entities in that dim
           dof_ids -- a dict mapping entity_id to the range of indices of dofs
                      associated to it.

        On a k-face for degree r, the dofs are given by the value of
           t^T u t
        evaluated at points enough to control P(r-k+1) for all the edge
        tangents of the face.
        `cell.make_points(entity_dim, entity_id, degree + 2)` happens to
        generate exactly those points needed.
        """
        dofs = []
        dof_ids = {}
        num_entities = len(cell.get_topology()[entity_dim])
        for entity_id in range(num_entities):
            pts = cell.make_points(entity_dim, entity_id, degree + 2)
            tangents = cell.compute_face_edge_tangents(entity_dim, entity_id)
            dofs += [InnerProduct(cell, t, t, pt)
                     for pt in pts
                     for t in tangents]
            num_new_dofs = len(pts) * len(tangents)
            dof_ids[entity_id] = list(range(offset, offset + num_new_dofs))
            offset += num_new_dofs
        return (dofs, dof_ids)


class Regge(CiarletElement):
    """The generalized Regge elements for symmetric-matrix-valued functions.
       REG(r) in dimension n is the space of polynomial symmetric-matrix-valued
       functions of degree r or less with tangential-tangential continuity.
    """
    def __init__(self, cell, degree):
        assert degree >= 0, "Regge start at degree 0!"
        # shape functions
        Ps = ONSymTensorPolynomialSet(cell, degree)
        # degrees of freedom
        Ls = ReggeDual(cell, degree)
        # mapping under affine transformation
        mapping = "double covariant piola"

        super(Regge, self).__init__(Ps, Ls, degree, mapping=mapping)
