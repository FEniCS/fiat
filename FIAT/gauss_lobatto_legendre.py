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

class GaussLobattoLegendreDualSet(dual_set.DualSet):
    """The dual basis for 1D continuous elements with nodes at the
    Gauss-Lobatto points."""
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [0], 1: [degree]},
                      1: {0: list(range(1, degree))}}
        lr = quadrature.GaussLobattoLegendreQuadratureLineRule(ref_el, degree+1)
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]

        super(GaussLobattoLegendreDualSet, self).__init__(nodes, ref_el, entity_ids)


class GaussLobattoLegendre(finite_element.FiniteElement):
    """1D continuous element with nodes at the Gauss-Lobatto points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Gauss-Lobatto-Legendre elements are only defined in one dimension.")
        dual = GaussLobattoLegendreDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        super(GaussLobattoLegendre, self).__init__(ref_el, dual, degree, formdegree)

    def tabulate(self, order, points, entity=None):
        dim = self.ref_el.get_dimension()
        if entity is None:
            entity = (dim, 0)

        entity_dim, entity_id = entity
        if entity_dim > dim:
            raise ValueError("entity dimension must be lower than ",dim)

        if entity_id == 0:
            xsrc = numpy.array([list(node.get_point_dict())[0][0] for node in self.dual.nodes])
        elif entity_id == 1:
            xsrc = numpy.array([point[0] for point in self.ref_el.vertices])
        else:
            raise ValueError("topological entity must be between 0 and 1")

        xdst = numpy.array(points).flatten()
        return barycentric_interpolation(xsrc, xdst, order)

    def value_shape(self):
        return ()

    def degree(self):
        return len(self.dual.nodes)-1

    def space_dimension(self):
        return len(self.dual.nodes)

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for GaussLobattoLegendre")

    def get_dual_set(self):
        raise NotImplementedError("get_dual_set is not implemented for GaussLobattoLegendre")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for GaussLobattoLegendre")

    def dmats(self):
        raise NotImplementedError("dmats not implemented for GaussLobattoLegendre")

    def get_num_members(self, arg):
        raise NotImplementedError("get_num_members not implemented for GaussLobattoLegendre")
