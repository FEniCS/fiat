# Copyright (C) 2018 Robert C. Kirby (Baylor University)
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

from FIAT import (expansions, polynomial_set, quadrature, dual_set,
                  finite_element, functional, Bubble, Lagrange)
#from expansions import polynomial_dimension as polydim
import numpy
from itertools import chain

polydim = expansions.polynomial_dimension

def WuXuH3Space(ref_el):
    """Constructs a basis for the the Wu Xu H^3 nonconforming space
    P^{(3,2)}(T) = P_3(T) + b_T P_1(T) + b_T^2 P_1(T),
    where b_T is the standard cubic bubble."""

    sd = ref_el.get_spatial_dimension()
    assert sd == 2

    em_deg = 7
    
    # Unfortunately,  b_T^2 P_1 has degree 7 (cubic squared times a linear)
    # so we need a high embedded degree!
    
    p7 = polynomial_set.ONPolynomialSet(ref_el, 7)

    dimp1 = polydim(ref_el, 1)
    dimp3 = polydim(ref_el, 3)
    dimp7 = polydim(ref_el, 7)
    
    # Here's the first bit we'll work with.  It's already expressed in terms
    # of the ON basis for P7, so we're golden.
    p3fromp7 = p7.take(list(range(dimp3)))

    # Rather than creating the barycentric coordinates ourself, let's
    # reuse the existing bubble functionality
    bT = Bubble(ref_el, 3)
    p1 = Lagrange(ref_el, 1)
    
    # next, we'll have to project b_T P1 and b_T^2 P1 onto P^7
    Q = quadrature.make_quadrature(ref_el, 8)
    Qpts = numpy.array(Q.get_points())
    Qwts = numpy.array(Q.get_weights())

    zero_index = tuple([0 for i in range(sd)])

    # it's just one bubble function: let's get a 1d array!
    bT_at_qpts = bT.tabulate(0, Qpts)[zero_index][0,:]
    p1_at_qpts = p1.tabulate(0, Qpts)[zero_index]

    # Note: difference in signature because bT, p1 are FE and p7 is a
    # polynomial set
    p7_at_qpts = p7.tabulate(Qpts)[zero_index]

    bubble_coeffs = numpy.zeros((6, dimp7), "d")

    # first three: bT P1, last three will be bT^2 P1
    foo = bT_at_qpts * p1_at_qpts * Qwts
    bubble_coeffs[:dimp1, :] = numpy.dot(foo, p7_at_qpts.T)

    foo = bT_at_qpts * foo
    bubble_coeffs[dimp1:2*dimp1, :] = numpy.dot(foo, p7_at_qpts.T)

    bubbles = polynomial_set.PolynomialSet(ref_el, 3, em_deg,
                                           p7.get_expansion_set(),
                                           bubble_coeffs,
                                           p7.get_dmats())

    return polynomial_set.polynomial_set_union_normalized(p3fromp7, bubbles)


class WuXuH3DualSet(dual_set.DualSet):
    """Dual basis for WuXu H3 nonconforming element consisting of
    vertex values and gradients and first and second normals at edge midpoints."""

    def __init__(self, ref_el):
        entity_ids = {}
        nodes = []
        cur = 0

        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        assert sd == 2

        pe = functional.PointEvaluation
        pd = functional.PointDerivative
        pnd = functional.PointNormalDerivative
        pndd = functional.PointNormalSecondDerivative
        
        # jet at each vertex

        entity_ids[0] = {}
        for v in sorted(top[0]):
            # point value
            nodes.append(pe(ref_el, verts[v]))
            # gradient
            for i in range(sd):
                alpha = [0]*sd
                alpha[i] = 1
                nodes.append(pd(ref_el, verts[v], alpha))

            entity_ids[0][v] = list(range(cur, cur+1+sd))
            cur += sd + 1

        entity_ids[1] = {}
        for e in sorted(top[1]):
            pt = ref_el.make_points(1, e, 2)[0]
            n = pnd(ref_el, e, pt)
            nn = pndd(ref_el, e, pt)
            nodes.extend([n, nn])
            entity_ids[1][e] = [cur, cur+1]
            cur += 2

        entity_ids[2] = {0: []}

        super(WuXuH3DualSet, self).__init__(nodes, ref_el, entity_ids)


class WuXuH3(finite_element.CiarletElement):
    """The Wu-Xu H3 finite element"""

    def __init__(self, ref_el):
        poly_set = WuXuH3Space(ref_el)
        dual = WuXuH3DualSet(ref_el)
        super(WuXuH3, self).__init__(poly_set, dual, 3)
