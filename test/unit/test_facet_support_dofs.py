# Copyright (C) 2016 Miklos Homolya
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

import pytest

import FIAT
from FIAT.reference_element import UFCInterval, UFCTriangle
from FIAT.finite_element import entity_support_dofs


@pytest.mark.parametrize(('base', 'extr', 'horiz_expected', 'vert_expected'),
                         [(("Discontinuous Lagrange", 0), ("Discontinuous Lagrange", 0),
                           {0: [0], 1: [0]},
                           {0: [0], 1: [0]}),
                          (("Discontinuous Lagrange", 1), ("Discontinuous Lagrange", 1),
                           {0: [0, 2], 1: [1, 3]},
                           {0: [0, 1], 1: [2, 3]}),
                          (("Lagrange", 1), ("Lagrange", 1),
                           {0: [0, 2], 1: [1, 3]},
                           {0: [0, 1], 1: [2, 3]}),
                          (("Discontinuous Lagrange", 0), ("Lagrange", 1),
                           {0: [0], 1: [1]},
                           {0: [0, 1], 1: [0, 1]}),
                          (("Lagrange", 1), ("Discontinuous Lagrange", 0),
                           {0: [0, 1], 1: [0, 1]},
                           {0: [0], 1: [1]})])
def test_quad(base, extr, horiz_expected, vert_expected):
    elem_A = FIAT.supported_elements[base[0]](UFCInterval(), base[1])
    elem_B = FIAT.supported_elements[extr[0]](UFCInterval(), extr[1])
    elem = FIAT.TensorProductElement(elem_A, elem_B)
    assert horiz_expected == entity_support_dofs(elem, (1, 0))
    assert vert_expected == entity_support_dofs(elem, (0, 1))


def test_quad_rtce():
    W0_h = FIAT.Lagrange(UFCInterval(), 1)
    W1_h = FIAT.DiscontinuousLagrange(UFCInterval(), 0)

    W0_v = FIAT.DiscontinuousLagrange(UFCInterval(), 0)
    W0 = FIAT.Hcurl(FIAT.TensorProductElement(W0_h, W0_v))

    W1_v = FIAT.Lagrange(UFCInterval(), 1)
    W1 = FIAT.Hcurl(FIAT.TensorProductElement(W1_h, W1_v))

    elem = FIAT.EnrichedElement(W0, W1)
    assert {0: [0, 1, 2], 1: [0, 1, 3]} == entity_support_dofs(elem, (1, 0))
    assert {0: [0, 2, 3], 1: [1, 2, 3]} == entity_support_dofs(elem, (0, 1))


def test_quad_rtcf():
    W0_h = FIAT.Lagrange(UFCInterval(), 1)
    W1_h = FIAT.DiscontinuousLagrange(UFCInterval(), 0)

    W0_v = FIAT.DiscontinuousLagrange(UFCInterval(), 0)
    W0 = FIAT.Hdiv(FIAT.TensorProductElement(W0_h, W0_v))

    W1_v = FIAT.Lagrange(UFCInterval(), 1)
    W1 = FIAT.Hdiv(FIAT.TensorProductElement(W1_h, W1_v))

    elem = FIAT.EnrichedElement(W0, W1)
    assert {0: [0, 1, 2], 1: [0, 1, 3]} == entity_support_dofs(elem, (1, 0))
    assert {0: [0, 2, 3], 1: [1, 2, 3]} == entity_support_dofs(elem, (0, 1))


@pytest.mark.parametrize(('base', 'extr', 'horiz_expected', 'vert_expected'),
                         [(("Discontinuous Lagrange", 0), ("Discontinuous Lagrange", 0),
                           {0: [0], 1: [0]},
                           {0: [0], 1: [0], 2: [0]}),
                          (("Discontinuous Lagrange", 1), ("Discontinuous Lagrange", 1),
                           {0: [0, 2, 4], 1: [1, 3, 5]},
                           {0: [2, 3, 4, 5], 1: [0, 1, 4, 5], 2: [0, 1, 2, 3]}),
                          (("Lagrange", 1), ("Lagrange", 1),
                           {0: [0, 2, 4], 1: [1, 3, 5]},
                           {0: [2, 3, 4, 5], 1: [0, 1, 4, 5], 2: [0, 1, 2, 3]}),
                          (("Discontinuous Lagrange", 0), ("Lagrange", 1),
                           {0: [0], 1: [1]},
                           {0: [0, 1], 1: [0, 1], 2: [0, 1]}),
                          (("Lagrange", 1), ("Discontinuous Lagrange", 0),
                           {0: [0, 1, 2], 1: [0, 1, 2]},
                           {0: [1, 2], 1: [0, 2], 2: [0, 1]})])
def test_prism(base, extr, horiz_expected, vert_expected):
    elem_A = FIAT.supported_elements[base[0]](UFCTriangle(), base[1])
    elem_B = FIAT.supported_elements[extr[0]](UFCInterval(), extr[1])
    elem = FIAT.TensorProductElement(elem_A, elem_B)
    assert horiz_expected == entity_support_dofs(elem, (2, 0))
    assert vert_expected == entity_support_dofs(elem, (1, 1))


@pytest.mark.parametrize(('space', 'degree', 'horiz_expected', 'vert_expected'),
                         [("Raviart-Thomas", 1,
                           {0: [0, 1, 2, 3], 1: [0, 1, 2, 4]},
                           {0: list(range(5)), 1: list(range(5)), 2: list(range(5))}),
                          ("Brezzi-Douglas-Marini", 1,
                           {0: [0, 1, 2, 3, 4, 5, 6], 1: [0, 1, 2, 3, 4, 5, 7]},
                           {0: list(range(8)), 1: list(range(8)), 2: list(range(8))})])
def test_prism_hdiv(space, degree, horiz_expected, vert_expected):
    W0_h = FIAT.supported_elements[space](UFCTriangle(), degree)
    W1_h = FIAT.DiscontinuousLagrange(UFCTriangle(), degree - 1)

    W0_v = FIAT.DiscontinuousLagrange(UFCInterval(), degree - 1)
    W0 = FIAT.Hdiv(FIAT.TensorProductElement(W0_h, W0_v))

    W1_v = FIAT.Lagrange(UFCInterval(), degree)
    W1 = FIAT.Hdiv(FIAT.TensorProductElement(W1_h, W1_v))

    elem = FIAT.EnrichedElement(W0, W1)
    assert horiz_expected == entity_support_dofs(elem, (2, 0))
    assert vert_expected == entity_support_dofs(elem, (1, 1))


@pytest.mark.parametrize(('space', 'degree', 'horiz_expected', 'vert_expected'),
                         [("Raviart-Thomas", 1,
                           {0: [0, 1, 2, 3, 5, 7], 1: [0, 1, 2, 4, 6, 8]},
                           {0: [1, 2] + list(range(3, 9)),
                            1: [0, 2] + list(range(3, 9)),
                            2: [0, 1] + list(range(3, 9))}),
                          ("Brezzi-Douglas-Marini", 1,
                           {0: list(range(3)) + list(range(3, 15, 2)),
                            1: list(range(3)) + list(range(4, 15, 2))},
                           {0: [1, 2] + list(range(3, 15)),
                            1: [0, 2] + list(range(3, 15)),
                            2: [0, 1] + list(range(3, 15))})])
def test_prism_hcurl(space, degree, horiz_expected, vert_expected):
    W0_h = FIAT.Lagrange(UFCTriangle(), degree)
    W1_h = FIAT.supported_elements[space](UFCTriangle(), degree)

    W0_v = FIAT.DiscontinuousLagrange(UFCInterval(), degree - 1)
    W0 = FIAT.Hcurl(FIAT.TensorProductElement(W0_h, W0_v))

    W1_v = FIAT.Lagrange(UFCInterval(), degree)
    W1 = FIAT.Hcurl(FIAT.TensorProductElement(W1_h, W1_v))

    elem = FIAT.EnrichedElement(W0, W1)
    assert horiz_expected == entity_support_dofs(elem, (2, 0))
    assert vert_expected == entity_support_dofs(elem, (1, 1))


def test_discontinuous_element():
    elem = FIAT.DiscontinuousElement(FIAT.Lagrange(UFCTriangle(), 3))
    assert entity_support_dofs(elem, 1) == {0: [1, 2, 3, 4],
                                            1: [0, 2, 5, 6],
                                            2: [0, 1, 7, 8]}


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
