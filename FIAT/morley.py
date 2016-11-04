# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
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

from FIAT import finite_element, polynomial_set, dual_set, functional


class MorleyDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points."""

    def __init__(self, ref_el):
        entity_ids = {}
        nodes = []
        cur = 0

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        if sd != 2:
            raise Exception("Illegal spatial dimension")

        # vertex point evaluations

        entity_ids[0] = {}
        for v in sorted(top[0]):
            nodes.append(functional.PointEvaluation(ref_el, verts[v]))

            entity_ids[0][v] = [cur]
            cur += 1

        # edge dof -- normal at each edge midpoint
        entity_ids[1] = {}
        for e in sorted(top[1]):
            pt = ref_el.make_points(1, e, 2)[0]
            n = functional.PointNormalDerivative(ref_el, e, pt)
            nodes.append(n)
            entity_ids[1][e] = [cur]
            cur += 1

        super(MorleyDualSet, self).__init__(nodes, ref_el, entity_ids)


class Morley(finite_element.CiarletElement):
    """The Morley finite element."""

    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 2)
        dual = MorleyDualSet(ref_el)
        super(Morley, self).__init__(poly_set, dual, 2)
