# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import numpy

from FIAT import finite_element, polynomial_set, dual_set, functional, quadrature
from FIAT.reference_element import LINE
from FIAT.barycentric_interpolation import barycentric_interpolation


class GaussLobattoLegendreDualSet(dual_set.DualSet):
    """The dual basis for 1D continuous elements with nodes at the
    Gauss-Lobatto points."""
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [0], 1: [degree]},
                      1: {0: list(range(1, degree))}}
        lr = quadrature.GaussLobattoLegendreQuadratureLineRule(ref_el, degree+1)
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]

        super(GaussLobattoLegendreDualSet, self).__init__(nodes, ref_el, entity_ids)


class GaussLobattoLegendre(finite_element.CiarletElement):
    """1D continuous element with nodes at the Gauss-Lobatto points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Gauss-Lobatto-Legendre elements are only defined in one dimension.")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = GaussLobattoLegendreDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        super(GaussLobattoLegendre, self).__init__(poly_set, dual, degree, formdegree)

    def tabulate(self, order, points, entity=None):
        # This overrides the default with a more numerically stable algorithm

        if entity is None:
            entity = (self.ref_el.get_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)

        xsrc = []
        for node in self.dual.nodes:
            # Assert singleton point for each node.
            (pt,), = node.get_point_dict().keys()
            xsrc.append(pt)
        xsrc = numpy.asarray(xsrc)
        xdst = numpy.array(list(map(transform, points))).flatten()
        return barycentric_interpolation(xsrc, xdst, order=order)
