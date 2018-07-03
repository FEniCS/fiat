# Copyright (C) 2018 Robert C. Kirby
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

# This is not quite Bell, but is 21-dofs and includes 3 extra constraint
# functionals.  The first 18 basis functions are the reference element
# bfs, but the extra three are used in the transformation theory.

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.reference_element import TRIANGLE, ufc_simplex


class BellDualSet(dual_set.DualSet):
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
            raise ValueError("Bell only defined on triangles")

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

        # we need an edge quadrature rule for the moment
        from FIAT.quadrature_schemes import create_quadrature
        from FIAT.jacobi import eval_jacobi
        rline = ufc_simplex(1)
        q1d = create_quadrature(rline, 8)
        q1dpts = q1d.get_points()
        leg4_at_qpts = eval_jacobi(0, 0, 4, 2.0*q1dpts - 1)

        imond = functional.IntegralMomentOfNormalDerivative
        entity_ids[1] = {}
        for e in sorted(top[1]):
            entity_ids[1][e] = [18+e]
            nodes.append(imond(ref_el, e, q1d, leg4_at_qpts))

        entity_ids[2] = {0: []}

        super(BellDualSet, self).__init__(nodes, ref_el, entity_ids)


class Bell(finite_element.CiarletElement):
    """The Bell finite element."""

    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 5)
        dual = BellDualSet(ref_el)
        super(Bell, self).__init__(poly_set, dual, 5)
