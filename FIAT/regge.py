# Copyright (C) 2015-2017 Lizao Li
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

import numpy

from FIAT.finite_element import FiniteElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import ONSymTensorPolynomialSet
from FIAT.functional import PointwiseInnerProductEvaluation as InnerProduct
from FIAT.functional import index_iterator


class ReggeDual(DualSet):

    def __init__(self, cell, degree):
        (dofs, ids) = self.generate_degrees_of_freedom(cell, degree)
        super(ReggeDual, self).__init__(dofs, cell, ids)

    def generate_degrees_of_freedom(self, cell, degree):
        """
        Suppose f is a k-face of the reference n-cell. Let t1,...,tk be a
        basis for the tangent space of f as n-vectors. Given a symmetric
        2-tensor field u on Rn. One set of dofs for Regge(r) on f is
        the moment of each of the (k+1)k/2 scalar functions
          [u(t1,t1),u(t1,t2),...,u(t1,tk),
           u(t2,t2), u(t2,t3),...,..., u(tk,tk)]
        aginst scalar polynomials of degrees (r-k+1). Here this is
        implemented as pointwise evaluations of those scalar functions.

        Below is an implementation for dimension 2--3. In dimension 1,
        Regge(r)=DG(r). It is awkward in the current FEniCS interface to
        implement the element uniformly for all dimensions due to its edge,
        facet=trig, cell style.
        """

        dofs = []
        ids = {}

        d = cell.get_spatial_dimension()
        if (d < 2) or (d > 3):
            raise ValueError("Regge elements only implemented for dimension 2--3.")

        # No vertex dof
        ids[0] = dict(list(zip(list(range(d + 1)), ([] for i in range(d + 1)))))
        # edge dofs
        (_dofs, _ids) = self._generate_edge_dofs(cell, degree, 0)
        dofs.extend(_dofs)
        ids[1] = _ids
        # facet dofs for 3D
        if d == 3:
            (_dofs, _ids) = self._generate_facet_dofs(cell, degree, len(dofs))
            dofs.extend(_dofs)
            ids[2] = _ids
        # Cell dofs
        (_dofs, _ids) = self._generate_cell_dofs(cell, degree, len(dofs))
        dofs.extend(_dofs)
        ids[d] = _ids
        return (dofs, ids)

    def _generate_edge_dofs(self, cell, degree, offset):
        """Generate dofs on edges."""
        dofs = []
        ids = {}
        for s in range(len(cell.get_topology()[1])):
            # Points to evaluate the inner product
            pts = cell.make_points(1, s, degree + 2)
            # Evalute squared length of the tagent vector along an edge
            t = cell.compute_edge_tangent(s)
            # Fill dofs
            dofs += [InnerProduct(cell, t, t, p) for p in pts]
            # Fill ids
            i = len(pts) * s
            ids[s] = list(range(offset + i, offset + i + len(pts)))
        return (dofs, ids)

    def _generate_facet_dofs(self, cell, degree, offset):
        """Generate dofs on facets in 3D."""
        # Return empty if there is no such dofs
        dofs = []
        ids = dict(list(zip(list(range(4)), ([] for i in range(4)))))
        if degree == 0:
            return (dofs, ids)
        # Compute dofs
        for s in range(len(cell.get_topology()[2])):
            # Points to evaluate the inner product
            pts = cell.make_points(2, s, degree + 2)
            # Let t1 and t2 be the two tangent vectors along a triangle
            # we evaluate u(t1,t1), u(t1,t2), u(t2,t2) at each point.
            (t1, t2) = cell.compute_face_tangents(s)
            # Fill dofs
            for p in pts:
                dofs += [InnerProduct(cell, t1, t1, p),
                         InnerProduct(cell, t1, t2, p),
                         InnerProduct(cell, t2, t2, p)]
            # Fill ids
            i = len(pts) * s * 3
            ids[s] = list(range(offset + i, offset + i + len(pts) * 3))
        return (dofs, ids)

    def _generate_cell_dofs(self, cell, degree, offset):
        """Generate dofs for cells."""
        # Return empty if there is no such dofs
        dofs = []
        d = cell.get_spatial_dimension()
        if (d == 2 and degree == 0) or (d == 3 and degree <= 1):
            return ([], {0: []})
        # Compute dofs. There is only one cell. So no need to loop here~
        # Points to evaluate the inner product
        pts = cell.make_points(d, 0, degree + 2)
        # Let {e1,..,ek} be the Euclidean basis. We evaluate inner products
        #   u(e1,e1), u(e1,e2), u(e1,e3), u(e2,e2), u(e2,e3),..., u(ek,ek)
        # at each point.
        e = numpy.eye(d)
        # Fill dofs
        for p in pts:
            dofs += [InnerProduct(cell, e[i], e[j], p)
                     for [i, j] in index_iterator((d, d)) if i <= j]
        # Fill ids
        ids = {0: list(range(offset, offset + len(pts) * d * (d + 1) // 2))}
        return (dofs, ids)


class Regge(FiniteElement):
    """
    The Regge elements on triangles and tetrahedra: the polynomial space
    described by the full polynomials of degree k with degrees of freedom
    to ensure its pullback as a metric to each interior facet and edge is
    single-valued.
    """

    def __init__(self, cell, degree):
        # Check degree
        assert degree >= 0, "Regge start at degree 0!"
        # Construct polynomial basis for d-vector fields
        Ps = ONSymTensorPolynomialSet(cell, degree)
        # Construct dual space
        Ls = ReggeDual(cell, degree)
        # Set mapping
        mapping = "pullback as metric"
        # Call init of super-class
        super(Regge, self).__init__(Ps, Ls, degree, mapping=mapping)
