# Copyright (C) 2020 Robert C Kirby (Baylor University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy

from FIAT import (
    BrezziDouglasMarini, Morley, QuinticArgyris, CubicHermite)

from FIAT.reference_element import (
    UFCTriangle,
    make_lattice)

from FIAT.pointwise_dual import compute_pointwise_dual as cpd

T = UFCTriangle()


@pytest.mark.parametrize("element",
                         [CubicHermite(T),
                          Morley(T),
                          QuinticArgyris(T),
                          BrezziDouglasMarini(T, 1, variant="integral")])
def test_pw_dual(element):
    deg = element.degree()
    ref_el = element.ref_el
    poly_set = element.poly_set
    pts = make_lattice(ref_el.vertices, deg)

    assert numpy.allclose(element.dual.to_riesz(poly_set),
                          cpd(element, pts).to_riesz(poly_set))
