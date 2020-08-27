# Copyright (C) 2020 Robert C. Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
import numpy as np
from FIAT.functional import Functional
from FIAT.dual_set import DualSet


def compute_pointwise_dual(el, pts):
    """Constructs a dual basis to the basis for el as a linear combination
    of a set of pointwise evaluations.  This is useful when the
    prescribed finite element isn't Ciarlet (e.g. the basis functions
    are provided explicitly as formulae).  Alternately, the element's
    given dual basis may involve differentiation, making run-time
    interpolation difficult in FIAT clients.  The pointwise dual,
    consisting only of pointwise evaluations, will effectively replace
    these derivatives with (automatically determined) finite
    differences.  This is exact on the polynomial space, but is an
    approximation if applied to functions outside the space.

    :param el: a :class:`FiniteElement`.
    :param pts: an iterable of points with the same length as el's
                dimension.  These points must be unisolvent for the
                polynomial space
    :returns: a :class `DualSet`
    """
    # We currently only have implemented this for scalar elements
    assert el.value_shape() == tuple()

    # We're only handling square systems:
    # This assertion needs to be generalized when we support vector-valued
    # elements.
    assert el.space_dimension() == len(pts)

    T = el.ref_el
    z = tuple([0] * T.get_dimension())

    V = el.tabulate(0, pts)[z]
    Vinv = np.linalg.inv(V)

    nds = []
    for i, coeffs in enumerate(Vinv.T):
        pt_dict = {pt: [(c, tuple())] for c, pt in zip(coeffs, pts) if np.abs(c) > 1.e-12}
        nds.append(Functional(T, (), pt_dict, {}, "node"))

    return DualSet(nds, T, el.entity_dofs())

