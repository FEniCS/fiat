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
    nbf = el.space_dimension()

    T = el.ref_el
    sd = T.get_dimension()

    assert np.asarray(pts).shape == (int(nbf / np.prod(el.value_shape())), sd)

    z = tuple([0] * sd)

    nds = []

    V = el.tabulate(0, pts)[z]

    # Make a square system, invert, and then put it back in the right
    # shape so we have (nbf, ..., npts) with more dimensions
    # for vector or tensor-valued elements.
    alphas = np.linalg.inv(V.reshape((nbf, -1)).T).reshape(V.shape)

    # Each row of alphas gives the coefficients of a functional,
    # represented, as elsewhere in FIAT, as a summation of
    # components of the input at particular points.

    for coeffs in alphas:
        pt_dict = {}
        # Iterates over the points themselves
        for k in range(coeffs.shape[-1]):
            lst = []
            # Iterates over the components of a vector- or tensor-
            # valued element
            for comp in np.ndindex(coeffs.shape[:-1]):
                blah = tuple(list(comp) + [k])
                # Drop coefficients that are close to zero
                if np.abs(coeffs[blah]) >= 1.e-12:
                    lst.append((coeffs[blah], comp))
            # Only add the point to the list if we actually got
            # a contribution in some component.
            if lst != []:
                pt_dict[pts[k]] = lst

        nds.append(Functional(T, el.value_shape(), pt_dict, {}, "node"))

    return DualSet(nds, T, el.entity_dofs())
