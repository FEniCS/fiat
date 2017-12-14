# Copyright (C) 2017 Miklos Homolya
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
import numpy as np

from FIAT import QuadratureElement, make_quadrature, ufc_simplex


@pytest.fixture(params=[1, 2, 3])
def cell(request):
    return ufc_simplex(request.param)


@pytest.fixture
def quadrature(cell):
    return make_quadrature(cell, 2)


@pytest.fixture
def element(cell, quadrature):
    return QuadratureElement(cell, quadrature.get_points())


def test_order(element, quadrature):
    with pytest.raises(ValueError):
        element.tabulate(1, quadrature.get_points())


def test_points(element, quadrature):
    points = quadrature.get_points()
    wrong_points = np.linspace(0.0, 1.0, points.size).reshape(points.shape)
    with pytest.raises(AssertionError):
        element.tabulate(0, wrong_points)


def test_entity(element, quadrature):
    dim = element.get_reference_element().get_spatial_dimension()
    points = make_quadrature(ufc_simplex(dim - 1), 2).get_points()
    with pytest.raises(ValueError):
        element.tabulate(0, points, entity=(dim - 1, 1))


def test_result(element, quadrature):
    dim = element.get_reference_element().get_spatial_dimension()
    points = quadrature.get_points()
    actual = element.tabulate(0, points)[(0,) * dim]
    assert np.allclose(np.eye(len(points)), actual)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
