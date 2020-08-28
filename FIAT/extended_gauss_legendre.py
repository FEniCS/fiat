# Copyright (C) 2019 Chris Eldred (Inria Grenoble Rhone-Alpes) and Werner Bauer (Inria Rennes)
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
# Written by Chris Eldred (Inria Grenoble Rhone-Alpes) and Werner Bauer (Inria Rennes) 2019

from FIAT import finite_element, polynomial_set, dual_set, functional, quadrature
from FIAT.reference_element import LINE


class ExtendedGaussLegendreDualSet(dual_set.DualSet):
    """The dual basis for 1D continuous elements with nodes at the
    Extended-Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [0], 1: [degree]},
                      1: {0: list(range(1, degree))}}
        lr = quadrature.ExtendedGaussLegendreQuadratureLineRule(ref_el, degree+1)
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]

        super(ExtendedGaussLegendreDualSet, self).__init__(nodes, ref_el, entity_ids)


class ExtendedGaussLegendre(finite_element.CiarletElement):
    """1D continuous element with nodes at the Extended-Gauss-Legendre points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("Extended-Gauss-Legendre elements are only defined in one dimension.")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = ExtendedGaussLegendreDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        super(ExtendedGaussLegendre, self).__init__(poly_set, dual, degree, formdegree)
