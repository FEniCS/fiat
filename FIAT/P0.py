# Copyright (C) 2005 The University of Chicago
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Robert C. Kirby
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

from FIAT import dual_set, functional, polynomial_set, finite_element
import numpy


class P0Dual(dual_set.DualSet):
    def __init__(self, ref_el):
        entity_ids = {}
        nodes = []
        vs = numpy.array(ref_el.get_vertices())
        if ref_el.get_dimension() == 0:
            bary = ()
        else:
            bary = tuple(numpy.average(vs, 0))

        nodes = [functional.PointEvaluation(ref_el, bary)]
        entity_ids = {}
        top = ref_el.get_topology()
        for dim in sorted(top):
            entity_ids[dim] = {}
            for entity in sorted(top[dim]):
                entity_ids[dim][entity] = []

        entity_ids[dim] = {0: [0]}

        super(P0Dual, self).__init__(nodes, ref_el, entity_ids)


class P0(finite_element.CiarletElement):
    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, 0)
        dual = P0Dual(ref_el)
        degree = 0
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(P0, self).__init__(poly_set, dual, degree, formdegree)
