# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
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
#
# Modified by Jan Blechta 2014

from __future__ import absolute_import, print_function, division

from FIAT import dual_set, finite_element, functional
from FIAT.raviart_thomas import RTSpace


class DRTDualSet(dual_set.DualSet):
    """Dual basis for Raviart-Thomas elements consisting of point
    evaluation of normals on facets of codimension 1 and internal
    moments against polynomials. This is the discontinuous version
    where all nodes are topologically associated with the cell itself"""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []

        sd = ref_el.get_spatial_dimension()
        t = ref_el.get_topology()

        # codimension 1 facets
        for i in range(len(t[sd - 1])):
            pts_cur = ref_el.make_points(sd - 1, i, sd + degree)
            for j in range(len(pts_cur)):
                pt_cur = pts_cur[j]
                f = functional.PointScaledNormalEvaluation(ref_el, i, pt_cur)
                nodes.append(f)

        # internal nodes.  Let's just use points at a lattice
        if degree > 0:
            cpe = functional.ComponentPointEvaluation
            pts = ref_el.make_points(sd, 0, degree + sd)
            for d in range(sd):
                for i in range(len(pts)):
                    l_cur = cpe(ref_el, d, (sd,), pts[i])
                    nodes.append(l_cur)

        # sets vertices (and in 3d, edges) to have no nodes
        for i in range(sd - 1):
            entity_ids[i] = {}
            for j in range(len(t[i])):
                entity_ids[i][j] = []

        # set codimension 1 (edges 2d, faces 3d) to have no dofs
        entity_ids[sd - 1] = {}
        for i in range(len(t[sd - 1])):
            entity_ids[sd - 1][i] = []

        # cell dofs
        entity_ids[sd] = {0: list(range(len(nodes)))}

        super(DRTDualSet, self).__init__(nodes, ref_el, entity_ids)


class DiscontinuousRaviartThomas(finite_element.CiarletElement):
    """The discontinuous Raviart-Thomas finite element"""

    def __init__(self, ref_el, q):

        degree = q - 1
        poly_set = RTSpace(ref_el, degree)
        dual = DRTDualSet(ref_el, degree)
        super(DiscontinuousRaviartThomas, self).__init__(poly_set, dual, degree,
                                                         mapping="contravariant piola")
