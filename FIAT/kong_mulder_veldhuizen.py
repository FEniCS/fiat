# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# contributions from Keith Roberts (University of Sao Paulo)
#
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from FIAT import (
    finite_element,
    dual_set,
    functional,
    bubble,
    lagrange,
    NodalEnrichedElement,
)
from FIAT.quadrature_schemes import create_quadrature  # noqa: F401


def _get_topology(ref_el, degree):
    """the topological association in a dictionary"""
    T = ref_el.topology
    sd = ref_el.get_spatial_dimension()
    if degree == 1:  # works for any spatial dimension.
        entity_ids = {0: dict((i, [i]) for i in range(len(T[0])))}
        for d in range(1, sd + 1):
            entity_ids[d] = dict((i, []) for i in range(len(T[d])))
    elif degree == 2:
        if sd == 2:
            entity_ids = {
                0: dict((i, [i]) for i in range(3)),
                1: dict((i, [i + 3]) for i in range(3)),
                2: {0: [6]},
            }
    elif degree == 3:
        if sd == 2:
            etop = [[3, 4], [5, 6], [7, 8]]
            entity_ids = {
                0: dict((i, [i]) for i in range(3)),
                1: dict((i, etop[i]) for i in range(3)),
                2: {0: [9, 10, 11]},
            }
    elif degree == 4:
        if sd == 2:
            etop = [[6, 3, 7], [8, 4, 9], [10, 5, 11]]
            entity_ids = {
                0: dict((i, [i]) for i in range(3)),
                1: dict((i, etop[i]) for i in range(3)),
                2: {0: [i for i in range(12, 18)]},
            }

    return entity_ids


def _enrich(ref_el, degree):
    """Pair spaces using bubbles following rules in ref listed below"""
    if degree == 1:
        return lagrange.Lagrange(ref_el, degree)
    elif degree < 5:
        P = lagrange.Lagrange(ref_el, degree)
        B = bubble.Bubble(ref_el, degree + 1)
    return NodalEnrichedElement(P, B)


class KongMulderVeldhuizenDualSet(dual_set.DualSet):
    """the dual basis for KMV simplical elements."""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        entity_ids = _get_topology(ref_el, degree)
        lr = create_quadrature(ref_el, degree, scheme="KMV")
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]
        super(KongMulderVeldhuizenDualSet, self).__init__(nodes, ref_el, entity_ids)


class KongMulderVeldhuizen(finite_element.CiarletElement):
    """The lumped finite element (NB: requires custom quad. "KMV" points
       to achieve a diagonal mass matrix).

       Ref: Higher-order triangular and tetrahedral finite elements with mass
       lumping for solving the wave equation
       M. J. S. CHIN-JOE-KONG, W. A. MULDER and M. VAN VELDHUIZEN
     """

    def __init__(self, ref_el, degree):
        if ref_el.shape != 2:
            raise NotImplementedError("Only triangles are currently implemented.")
        if degree > 4:
            raise NotImplementedError("Only P < 5 are currently implemented.")
        S = _enrich(ref_el, degree)
        poly_set = S.get_nodal_basis()
        dual = KongMulderVeldhuizenDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        if degree == 1:
            super(KongMulderVeldhuizen, self).__init__(
                poly_set, dual, 1, formdegree
            )
        elif degree == 2:
            super(KongMulderVeldhuizen, self).__init__(
                poly_set, dual, 3 , formdegree
            )
        # Bubble elements for cubic KMV have a dimension of 12 (but NodalEnrichedElement has a dimension of 13)
        elif degree == 3:
            super(KongMulderVeldhuizen, self).__init__(
                poly_set.take(range(12)), dual, 4, formdegree
            )
        # Bubble elements for cubic KMV have a dimension of 18 (but NodalEnrichedElement has a dimension of 21)
        elif degree == 4:
            super(KongMulderVeldhuizen, self).__init__(
                poly_set.take(range(18)), dual, 5, formdegree
            )
