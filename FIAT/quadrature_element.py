# -*- coding: utf-8 -*-

# Copyright (C) 2007-2016 Kristian B. Oelgaard
#
# This file is part of FFC.
#
# FFC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FFC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FFC. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Garth N. Wells 2006-2009

# Python modules.
import numpy
import six

# FFC modules.
from ffc.log import error, info_red

# FIAT modules.
from FIAT.dual_set import DualSet
from FIAT.finite_element import FiniteElement
from FIAT.functional import PointEvaluation


class QuadratureElement(FiniteElement):

    """Write description of QuadratureElement"""

    def __init__(self, ref_el, points):
        "Create QuadratureElement"

        # Create entity dofs.
        entity_dofs = {dim: {entity: [] for entity in entities}
                       for dim, entities in six.iteritems(ref_el.get_topology())}
        entity_dofs[ref_el.get_dimension()] = {0: list(range(len(points)))}

        # The dual is a simply the PointEvaluation at the quadrature points
        # FIXME: KBO: Check if this gives expected results for code like evaluate_dof.
        nodes = [PointEvaluation(ref_el, tuple(point)) for point in points]

        # Construct the dual set
        dual = DualSet(nodes, ref_el, entity_dofs)

        super(QuadratureElement, self).__init__(ref_el, dual, order=None)
        self._points = points  # save the quadrature points

    def value_shape(self):
        "The QuadratureElement is scalar valued"
        return ()

    def tabulate(self, order, points):
        """Return the identity matrix of size (num_quad_points, num_quad_points),
        in a format that monomialintegration and monomialtabulation understands."""

        # Derivatives are not defined on a QuadratureElement
        # FIXME: currently this check results in a warning (should be RuntimeError)
        # because otherwise some forms fails if QuadratureElement is used in a
        # mixed element e.g.,
        # element = CG + QuadratureElement
        # (v, w) = BasisFunctions(element)
        # grad(w): this is in error and should result in a runtime error
        # grad(v): this should be OK, but currently will raise a warning because
        # derivatives are tabulated for ALL elements in the mixed element.
        # This issue should be fixed in UFL and then we can switch on the
        # RuntimeError again.
        if order:
            # error("Derivatives are not defined on a QuadratureElement")
            info_red("\n*** WARNING: Derivatives are not defined on a QuadratureElement,")
            info_red("             returning values of basisfunction.\n")

        # Check that incoming points are equal to the quadrature points.
        if len(points) != len(self._points) or abs(numpy.array(points) - self._points).max() > 1e-12:
            print("\npoints:\n", numpy.array(points))
            print("\nquad points:\n", self._points)
            error("Points must be equal to coordinates of quadrature points")

        # Return the identity matrix of size len(self._points) in a
        # suitable format for tensor and quadrature representations.
        values = numpy.eye(len(self._points))
        dim = self.ref_el.get_spatial_dimension()
        return {(0,) * dim: values}


def _create_entity_dofs(fiat_cell, num_dofs):
    "This function is ripped from FIAT/discontinuous_lagrange.py"
    entity_dofs = {}
    top = fiat_cell.get_topology()
    for dim in sorted(top):
        entity_dofs[dim] = {}
        for entity in sorted(top[dim]):
            entity_dofs[dim][entity] = []
    entity_dofs[dim][0] = list(range(num_dofs))
    return entity_dofs
