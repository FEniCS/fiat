# -*- coding: utf-8 -*-

# Copyright (C) 2007-2016 Kristian B. Oelgaard
# Copyright (C) 2017 Miklós Homolya
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Garth N. Wells 2006-2009

import numpy

from FIAT.dual_set import DualSet
from FIAT.finite_element import FiniteElement
from FIAT.functional import PointEvaluation


class QuadratureElement(FiniteElement):
    """A set of quadrature points pretending to be a finite element."""

    def __init__(self, ref_el, points, weights=None):
        # Create entity dofs.
        entity_dofs = {dim: {entity: [] for entity in entities}
                       for dim, entities in ref_el.get_topology().items()}
        entity_dofs[ref_el.get_dimension()] = {0: list(range(len(points)))}

        # The dual nodes are PointEvaluations at the quadrature points.
        # FIXME: KBO: Check if this gives expected results for code like evaluate_dof.
        nodes = [PointEvaluation(ref_el, tuple(point)) for point in points]

        # Construct the dual set
        dual = DualSet(nodes, ref_el, entity_dofs)

        super(QuadratureElement, self).__init__(ref_el, dual, order=None)
        self._points = points  # save the quadrature points & weights
        self._weights = weights

    def value_shape(self):
        "The QuadratureElement is scalar valued"
        return ()

    def tabulate(self, order, points, entity=None):
        """Return the identity matrix of size (num_quad_points, num_quad_points),
        in a format that monomialintegration and monomialtabulation understands."""

        if entity is not None and entity != (self.ref_el.get_dimension(), 0):
            raise ValueError('QuadratureElement does not "tabulate" on subentities.')

        # Derivatives are not defined on a QuadratureElement
        if order:
            raise ValueError("Derivatives are not defined on a QuadratureElement.")

        # Check that incoming points are equal to the quadrature points.
        if len(points) != len(self._points) or abs(numpy.array(points) - self._points).max() > 1e-12:
            raise AssertionError("Mismatch of quadrature points!")

        # Return the identity matrix of size len(self._points).
        values = numpy.eye(len(self._points))
        dim = self.ref_el.get_spatial_dimension()
        return {(0,) * dim: values}

    @staticmethod
    def is_nodal():
        # No polynomial basis, but still nodal.
        return True
