# Copyright (C) 2020 Robert C. Kirby (Baylor University)
#
# contributions by Keith Roberts (University of São Paulo)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from FIAT import (
    finite_element,
    dual_set,
    functional,
    Bubble,
    FacetBubble,
    Lagrange,
    NodalEnrichedElement,
    RestrictedElement,
    reference_element,
)
from FIAT.quadrature_schemes import create_quadrature

TRIANGLE = reference_element.UFCTriangle()
TETRAHEDRON = reference_element.UFCTetrahedron()


def _get_entity_ids(ref_el, degree):
    """The topological association in a dictionary"""
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
        elif sd == 3:
            entity_ids = {
                0: dict((i, [i]) for i in range(4)),
                1: dict((i, [i + 4]) for i in range(6)),
                2: dict((i, [i + 10]) for i in range(4)),
                3: {0: [14]},
            }
    elif degree == 3:
        if sd == 2:
            etop = [[3, 4], [6, 5], [7, 8]]
            entity_ids = {
                0: dict((i, [i]) for i in range(3)),
                1: dict((i, etop[i]) for i in range(3)),
                2: {0: [9, 10, 11]},
            }
        elif sd == 3:
            etop = [[4, 5], [7, 6], [8, 9], [11, 10], [12, 13], [14, 15]]
            ftop = [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27]]
            entity_ids = {
                0: dict((i, [i]) for i in range(4)),
                1: dict((i, etop[i]) for i in range(6)),
                2: dict((i, ftop[i]) for i in range(4)),
                3: {0: [28, 29, 30, 31]},
            }
    elif degree == 4:
        if sd == 2:
            etop = [[6, 3, 7], [9, 4, 8], [10, 5, 11]]
            entity_ids = {
                0: dict((i, [i]) for i in range(3)),
                1: dict((i, etop[i]) for i in range(3)),
                2: {0: [i for i in range(12, 18)]},
            }
    elif degree == 5:
        if sd == 2:
            etop = [[9, 3, 4, 10], [12, 6, 5, 11], [13, 7, 8, 14]]
            entity_ids = {
                0: dict((i, [i]) for i in range(3)),
                1: dict((i, etop[i]) for i in range(3)),
                2: {0: [i for i in range(15, 30)]},
            }
    return entity_ids


def bump(T, deg):
    """Increase degree of polynomial along face/edges"""
    sd = T.get_spatial_dimension()
    if deg == 1:
        return (0, 0)
    else:
        if sd == 2:
            if deg < 5:
                return (1, 1)
            elif deg == 5:
                return (2, 2)
            else:
                raise ValueError("Degree not supported")
        elif sd == 3:
            if deg < 4:
                return (1, 2)
            else:
                raise ValueError("Degree not supported")
        else:
            raise ValueError("Dimension of element is not supported")


def KongMulderVeldhuizenSpace(T, deg):
    sd = T.get_spatial_dimension()
    if deg == 1:
        return Lagrange(T, 1).poly_set
    else:
        L = Lagrange(T, deg)
        # Toss the bubble from Lagrange since it's dependent
        # on the higher-dimensional bubbles
        if sd == 2:
            inds = [
                i
                for i in range(L.space_dimension())
                if i not in L.dual.entity_ids[sd][0]
            ]
        elif sd == 3:
            not_inds = [L.dual.entity_ids[sd][0]] + [
                L.dual.entity_ids[sd - 1][f] for f in L.dual.entity_ids[sd - 1]
            ]
            not_inds = [item for sublist in not_inds for item in sublist]
            inds = [i for i in range(L.space_dimension()) if i not in not_inds]
        RL = RestrictedElement(L, inds)
        # interior cell bubble
        bubs = Bubble(T, deg + bump(T, deg)[1])
        if sd == 2:
            return NodalEnrichedElement(RL, bubs).poly_set
        elif sd == 3:
            # bubble on the facet
            fbubs = FacetBubble(T, deg + bump(T, deg)[0])
            return NodalEnrichedElement(RL, bubs, fbubs).poly_set


class KongMulderVeldhuizenDualSet(dual_set.DualSet):
    """The dual basis for KMV simplical elements."""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        entity_ids = _get_entity_ids(ref_el, degree)
        lr = create_quadrature(ref_el, degree, scheme="KMV")
        nodes = [functional.PointEvaluation(ref_el, x) for x in lr.pts]
        super(KongMulderVeldhuizenDualSet, self).__init__(nodes, ref_el, entity_ids)


class KongMulderVeldhuizen(finite_element.CiarletElement):
    """The "lumped" simplical finite element (NB: requires custom quad. "KMV" points to achieve a diagonal mass matrix).

    References
    ----------

    Higher-order triangular and tetrahedral finite elements with mass
    lumping for solving the wave equation
    M. J. S. CHIN-JOE-KONG, W. A. MULDER and M. VAN VELDHUIZEN

    HIGHER-ORDER MASS-LUMPED FINITE ELEMENTS FOR THE WAVE EQUATION
    W.A. MULDER

    NEW HIGHER-ORDER MASS-LUMPED TETRAHEDRAL ELEMENTS
    S. GEEVERS, W.A. MULDER, AND J.J.W. VAN DER VEGT

     """

    def __init__(self, ref_el, degree):
        if ref_el != TRIANGLE and ref_el != TETRAHEDRON:
            raise ValueError("KMV is only valid for triangles and tetrahedrals")
        if degree > 5 and ref_el == TRIANGLE:
            raise NotImplementedError("Only P < 6 for triangles are implemented.")
        if degree > 3 and ref_el == TETRAHEDRON:
            raise NotImplementedError("Only P < 4 for tetrahedrals are implemented.")
        S = KongMulderVeldhuizenSpace(ref_el, degree)

        dual = KongMulderVeldhuizenDualSet(ref_el, degree)
        formdegree = 0  # 0-form
        super(KongMulderVeldhuizen, self).__init__(
            S, dual, degree + max(bump(ref_el, degree)), formdegree
        )
