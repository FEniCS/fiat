# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Colin Cotter (Imperial College London)
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

from FIAT import finite_element, polynomial_set, dual_set, functional, P0, quadrature
from FIAT.reference_element import ufc_simplex
import numpy


class DiscontinuousTaylorDualSet(dual_set.DualSet):
    """The dual basis for Taylor elements.  This class works for
    intervals.  Nodes are function and derivative evaluation
    at the midpoint. This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""

    def __init__(self, ref_el, degree):

        assert(ref_el.get_spatial_dimension() == 1)

        entity_ids = {}
        nodes = []

        Q = quadrature.make_quadrature(ref_el, 2 * (degree + 1))

        f_at_qpts = numpy.ones(len(Q.wts))
        nodes.append(functional.IntegralMoment(ref_el, Q, f_at_qpts))

        vertices = ref_el.get_vertices()
        midpoint = (vertices[1][0] + vertices[0][0]) / 2.0
        for k in range(1, degree+1):
            nodes.append(functional.PointDerivative(ref_el, (midpoint,), [k]))

        entity_ids[0] = {}
        entity_ids[1] = {}
        entity_ids[0][0] = []
        entity_ids[0][1] = []
        entity_ids[1][0] = list(range(degree+1))

        dual_set.DualSet.__init__(self, nodes, ref_el, entity_ids)


class HigherOrderDiscontinuousTaylor(finite_element.FiniteElement):
    """The discontinuous Taylor finite element. Use a Taylor basis for DG."""

    def __init__(self, ref_el, degree):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = DiscontinuousTaylorDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        finite_element.FiniteElement.__init__(self, poly_set, dual, degree, formdegree)


def DiscontinuousTaylor(ref_el, degree):
    if degree == 0:
        return P0.P0(ref_el)
    else:
        return HigherOrderDiscontinuousTaylor(ref_el, degree)

if __name__ == "__main__":

    T = ufc_simplex(1)
    element = DiscontinuousTaylor(T, 1)
    pts = [(0.0,), (0.5,), (1.0,)]
    a = element.tabulate(1, pts)

    assert(numpy.abs(a[0, ]-numpy.array([[1., 1., 1.],
                                         [-0.5, 0., 0.5]])).max() < 1.0e-10)
    assert(numpy.abs(a[1, ]-numpy.array([[0., 0., 0.],
                                         [1.0, 1.0, 1.0]])).max() < 1.0e-10)
