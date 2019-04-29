# Copyright (C) 2018 Cyrus Cheng (Imperial College London)
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
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2018

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.reference_element import (Point,
                                    DefaultLine,
                                    UFCInterval,
                                    UFCQuadrilateral,
                                    UFCHexahedron,
                                    UFCTriangle,
                                    UFCTetrahedron,
                                    make_affine_mapping)
from FIAT.P0 import P0Dual
import numpy as np

hypercube_simplex_map = {Point(): Point(),
                         DefaultLine(): DefaultLine(),
                         UFCInterval(): UFCInterval(),
                         UFCQuadrilateral(): UFCTriangle(),
                         UFCHexahedron(): UFCTetrahedron()}


class DPC0(finite_element.CiarletElement):
    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(hypercube_simplex_map[ref_el], 0)
        dual = P0Dual(ref_el)
        degree = 0
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(DPC0, self).__init__(poly_set=poly_set,
                                   dual=dual,
                                   order=degree,
                                   ref_el=ref_el,
                                   formdegree=formdegree)


class DPCDualSet(dual_set.DualSet):
    """The dual basis for DPC elements.  This class works for
    hypercubes of any dimension.  Nodes are point evaluation at
    equispaced points.  This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []

        # Change coordinates here.
        # Vertices of the simplex corresponding to the reference element.
        v_simplex = hypercube_simplex_map[ref_el].get_vertices()
        # Vertices of the reference element.
        v_hypercube = ref_el.get_vertices()
        # For the mapping, first two vertices are unchanged in all dimensions.
        v_ = [v_hypercube[0], v_hypercube[int(-0.5*len(v_hypercube))]]

        # For dimension 1 upwards,
        # take the next vertex and map it to the midpoint of the edge/face it belongs to, and shares
        # with no other points.
        for d in range(1, ref_el.get_dimension()):
            v_.append(tuple(np.asarray(v_hypercube[ref_el.get_dimension() - d] +
                            np.average(np.asarray(v_hypercube[::2]), axis=0))))
        A, b = make_affine_mapping(v_simplex, tuple(v_))  # Make affine mapping to be used later.

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = hypercube_simplex_map[ref_el].get_topology()
        cube_topology = ref_el.get_topology()

        cur = 0
        for dim in sorted(top):
            entity_ids[dim] = {}
            for entity in sorted(top[dim]):
                pts_cur = hypercube_simplex_map[ref_el].make_points(dim, entity, degree)
                pts_cur = [tuple(np.matmul(A, np.array(x)) + b) for x in pts_cur]
                nodes_cur = [functional.PointEvaluation(ref_el, x)
                             for x in pts_cur]
                nnodes_cur = len(nodes_cur)
                nodes += nodes_cur
                cur += nnodes_cur
            for entity in sorted(cube_topology[dim]):
                entity_ids[dim][entity] = []

        entity_ids[dim][0] = list(range(len(nodes)))
        super(DPCDualSet, self).__init__(nodes, ref_el, entity_ids)


class HigherOrderDPC(finite_element.CiarletElement):
    """The DPC finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        poly_set = polynomial_set.ONPolynomialSet(hypercube_simplex_map[ref_el], degree)
        dual = DPCDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(HigherOrderDPC, self).__init__(poly_set=poly_set,
                                             dual=dual,
                                             order=degree,
                                             ref_el=ref_el,
                                             formdegree=formdegree)


def DPC(ref_el, degree):
    if degree == 0:
        return DPC0(ref_el)
    else:
        return HigherOrderDPC(ref_el, degree)
