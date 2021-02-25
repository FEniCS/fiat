# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2020

import numpy

from FIAT import finite_element, dual_set, functional, quadrature
from FIAT.reference_element import LINE
from FIAT.barycentric_interpolation import barycentric_interpolation


class GaussLegendreDualSet(dual_set.DualSet):
    """The dual basis for 1D discontinuous elements with nodes at the
    Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [], 1: []},
                      1: {0: list(range(0, degree+1))}}
        lr = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]

        super(GaussLegendreDualSet, self).__init__(nodes, ref_el, entity_ids)


class GaussLegendre(finite_element.FiniteElement):
    """1D discontinuous element with nodes at the Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Gauss-Legendre elements are only defined in one dimension.")
        dual = GaussLegendreDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(GaussLegendre, self).__init__(ref_el, dual, degree, formdegree)

    def tabulate(self, order, points, entity=None):

        if entity is None:
            entity = (self.ref_el.get_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        points = list(map(transform, points))

        xsrc = numpy.array([list(node.get_point_dict())[0][0] for node in self.dual.nodes])
        xdst = numpy.array(points).flatten()
        return barycentric_interpolation(xsrc, xdst, order)

    @staticmethod
    def is_nodal():
        return True

    def value_shape(self):
        return ()

    def degree(self):
        return self.order

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for GaussLegendre")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for GaussLegendre")

    def dmats(self):
        raise NotImplementedError("dmats not implemented for GaussLegendre")

    def get_num_members(self, arg):
        raise NotImplementedError("get_num_members not implemented for GaussLegendre")
