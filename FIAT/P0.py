# Copyright (C) 2005 The University of Chicago
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
# Written by Robert C. Kirby
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

from __future__ import absolute_import, print_function, division

from FIAT import dual_set, functional, polynomial_set, finite_element
import numpy


class P0Dual(dual_set.DualSet):
    def __init__(self, ref_el):
        entity_ids = {}
        nodes = []
        vs = numpy.array(ref_el.get_vertices())
        bary = tuple(numpy.average(vs, 0))

        nodes = [functional.PointEvaluation(ref_el, bary)]
        entity_ids = {}
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        for dim in sorted(top):
            entity_ids[dim] = {}
            for entity in sorted(top[dim]):
                entity_ids[dim][entity] = []

        entity_ids[sd] = {0: [0]}

        super(P0Dual, self).__init__(nodes, ref_el, entity_ids)


class P0(finite_element.CiarletElement):
    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 0)
        dual = P0Dual(ref_el)
        degree = 0
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(P0, self).__init__(poly_set, dual, degree, formdegree)
