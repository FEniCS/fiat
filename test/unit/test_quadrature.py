# Copyright (C) 2015 Imperial College London and others.
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
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015

from __future__ import absolute_import
from __future__ import division

import numpy
import pytest
import FIAT
from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import FiredrakeQuadrilateral, TensorProductCell


@pytest.fixture(scope='module')
def interval():
    return UFCInterval()


@pytest.fixture(scope='module')
def triangle():
    return UFCTriangle()


@pytest.fixture(scope='module')
def tetrahedron():
    return UFCTetrahedron()


@pytest.fixture(scope='module')
def quadrilateral():
    return FiredrakeQuadrilateral()


@pytest.fixture(scope='module')
def extr_interval():
    """Extruded interval = interval x interval"""
    return TensorProductCell(UFCInterval(), UFCInterval())


@pytest.fixture(scope='module')
def extr_triangle():
    """Extruded triangle = triangle x interval"""
    return TensorProductCell(UFCTriangle(), UFCInterval())


@pytest.fixture(scope='module')
def extr_quadrilateral():
    """Extruded quadrilateral = quadrilateral x interval"""
    return TensorProductCell(FiredrakeQuadrilateral(), UFCInterval())


@pytest.fixture(params=["canonical", "default"])
def scheme(request):
    return request.param


@pytest.mark.parametrize("degree", range(8))
def test_create_quadrature_interval(interval, degree, scheme):
    q = FIAT.create_quadrature(interval, degree, scheme)
    assert numpy.allclose(q.integrate(lambda x: x[0]**degree), 1/(degree + 1))


@pytest.mark.parametrize("degree", range(8))
def test_create_quadrature_triangle(triangle, degree, scheme):
    q = FIAT.create_quadrature(triangle, degree, scheme)
    assert numpy.allclose(q.integrate(lambda x: sum(x)**degree), 1/(degree + 2))


@pytest.mark.parametrize("degree", range(8))
def test_create_quadrature_tetrahedron(tetrahedron, degree, scheme):
    q = FIAT.create_quadrature(tetrahedron, degree, scheme)
    assert numpy.allclose(q.integrate(lambda x: sum(x)**degree), 1/(2*degree + 6))


@pytest.mark.parametrize("extrdeg", range(4))
@pytest.mark.parametrize("basedeg", range(5))
def test_create_quadrature_extr_interval(extr_interval, basedeg, extrdeg, scheme):
    q = FIAT.create_quadrature(extr_interval, (basedeg, extrdeg), scheme)
    assert numpy.allclose(q.integrate(lambda (x, y): x**basedeg * y**extrdeg),
                          1/(basedeg + 1) * 1/(extrdeg + 1))


@pytest.mark.parametrize("extrdeg", range(4))
@pytest.mark.parametrize("basedeg", range(5))
def test_create_quadrature_extr_triangle(extr_triangle, basedeg, extrdeg, scheme):
    q = FIAT.create_quadrature(extr_triangle, (basedeg, extrdeg), scheme)
    assert numpy.allclose(q.integrate(lambda (x, y, z): (x + y)**basedeg * z**extrdeg),
                          1/(basedeg + 2) * 1/(extrdeg + 1))


@pytest.mark.parametrize("degree", range(8))
def test_create_quadrature_quadrilateral(quadrilateral, degree, scheme):
    q = FIAT.create_quadrature(quadrilateral, degree, scheme)
    assert numpy.allclose(q.integrate(lambda x: sum(x)**degree),
                          (2**(degree + 2) - 2) / ((degree + 1)*(degree + 2)))


@pytest.mark.parametrize("extrdeg", range(4))
@pytest.mark.parametrize("basedeg", range(5))
def test_create_quadrature_extr_quadrilateral(extr_quadrilateral, basedeg, extrdeg, scheme):
    q = FIAT.create_quadrature(extr_quadrilateral, (basedeg, extrdeg), scheme)
    assert numpy.allclose(q.integrate(lambda (x, y, z): (x + y)**basedeg * z**extrdeg),
                          (2**(basedeg + 2) - 2) / ((basedeg + 1)*(basedeg + 2)) * 1/(extrdeg + 1))


@pytest.mark.parametrize("cell", [interval(),
                                  triangle(),
                                  tetrahedron(),
                                  quadrilateral()])
def test_invalid_quadrature_degree(cell, scheme):
    with pytest.raises(ValueError):
        FIAT.create_quadrature(cell, -1, scheme)


@pytest.mark.parametrize("cell", [extr_interval(),
                                  extr_triangle(),
                                  extr_quadrilateral()])
def test_invalid_quadrature_degree_tensor_prod(cell):
    with pytest.raises(ValueError):
        FIAT.create_quadrature(cell, (-1, -1))


@pytest.mark.parametrize("cell", [interval(),
                                  triangle(),
                                  tetrahedron(),
                                  quadrilateral()])
def test_high_degree_runtime_error(cell):
    with pytest.raises(RuntimeError):
        FIAT.create_quadrature(cell, 60)


@pytest.mark.parametrize("cell", [extr_interval(),
                                  extr_triangle(),
                                  extr_quadrilateral()])
def test_high_degree_runtime_error_tensor_prod(cell):
    with pytest.raises(RuntimeError):
        FIAT.create_quadrature(cell, (60, 60))


@pytest.mark.parametrize(("points, degree"), ((p, d)
                                              for p in range(2, 10)
                                              for d in range(2*p - 2)))
def test_gauss_lobatto_legendre_quadrature(interval, points, degree):
    """Check that the quadrature rules correctly integrate all the right
    polynomial degrees."""

    q = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule(interval, points)

    assert numpy.round(q.integrate(lambda x: x[0]**degree) - 1./(degree+1), 14) == 0.


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
