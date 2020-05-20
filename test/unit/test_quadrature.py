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

import numpy
import pytest
import FIAT
from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import UFCQuadrilateral, UFCHexahedron, TensorProductCell


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
    return UFCQuadrilateral()


@pytest.fixture(scope='module')
def hexahedron():
    return UFCHexahedron()


# This unified fixture enables tests parametrised over different cells.
@pytest.fixture(params=["interval",
                        "triangle",
                        "quadrilateral",
                        "hexahedron"])
def cell(request):
    if request.param == "interval":
        return UFCInterval()
    elif request.param == "triangle":
        return UFCTriangle()
    elif request.param == "quadrilateral":
        return UFCTriangle()
    elif request.param == "hexahedron":
        return UFCTriangle()


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
    return TensorProductCell(UFCQuadrilateral(), UFCInterval())


# This unified fixture enables tests parametrised over different extruded cells.
@pytest.fixture(params=["extr_interval",
                        "extr_triangle",
                        "extr_quadrilateral"])
def extr_cell(request):
    if request.param == "extr_interval":
        return TensorProductCell(UFCInterval(), UFCInterval())
    elif request.param == "extr_triangle":
        return TensorProductCell(UFCTriangle(), UFCInterval())
    elif request.param == "extr_quadrilateral":
        return TensorProductCell(UFCQuadrilateral(), UFCInterval())


@pytest.fixture(params=["canonical", "default"])
def scheme(request):
    return request.param


def test_invalid_quadrature_rule():
    from FIAT.quadrature import QuadratureRule
    with pytest.raises(ValueError):
        QuadratureRule(UFCInterval(), [[0.5, 0.5]], [0.5, 0.5, 0.5])


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
    assert numpy.allclose(q.integrate(lambda x: x[0]**basedeg * x[1]**extrdeg),
                          1/(basedeg + 1) * 1/(extrdeg + 1))


@pytest.mark.parametrize("extrdeg", range(4))
@pytest.mark.parametrize("basedeg", range(5))
def test_create_quadrature_extr_triangle(extr_triangle, basedeg, extrdeg, scheme):
    q = FIAT.create_quadrature(extr_triangle, (basedeg, extrdeg), scheme)
    assert numpy.allclose(q.integrate(lambda x: (x[0] + x[1])**basedeg * x[2]**extrdeg),
                          1/(basedeg + 2) * 1/(extrdeg + 1))


@pytest.mark.parametrize("degree", range(8))
def test_create_quadrature_quadrilateral(quadrilateral, degree, scheme):
    q = FIAT.create_quadrature(quadrilateral, degree, scheme)
    assert numpy.allclose(q.integrate(lambda x: sum(x)**degree),
                          (2**(degree + 2) - 2) / ((degree + 1)*(degree + 2)))


@pytest.mark.parametrize("degree", range(8))
def test_create_quadrature_hexahedron(hexahedron, degree, scheme):
    q = FIAT.create_quadrature(hexahedron, degree, scheme)
    assert numpy.allclose(q.integrate(lambda x: sum(x)**degree),
                          -3 * (2**(degree + 3) - 3**(degree + 2) - 1) / ((degree + 1)*(degree + 2)*(degree + 3)))


@pytest.mark.parametrize("extrdeg", range(4))
@pytest.mark.parametrize("basedeg", range(5))
def test_create_quadrature_extr_quadrilateral(extr_quadrilateral, basedeg, extrdeg, scheme):
    q = FIAT.create_quadrature(extr_quadrilateral, (basedeg, extrdeg), scheme)
    assert numpy.allclose(q.integrate(lambda x: (x[0] + x[1])**basedeg * x[2]**extrdeg),
                          (2**(basedeg + 2) - 2) / ((basedeg + 1)*(basedeg + 2)) * 1/(extrdeg + 1))


def test_invalid_quadrature_degree(cell, scheme):
    with pytest.raises(ValueError):
        FIAT.create_quadrature(cell, -1, scheme)


def test_invalid_quadrature_degree_tensor_prod(extr_cell):
    with pytest.raises(ValueError):
        FIAT.create_quadrature(extr_cell, (-1, -1))


def test_tensor_product_composition(interval, triangle, extr_triangle, scheme):
    degree = (4, 4)
    qa = FIAT.create_quadrature(triangle, degree[0], scheme)
    qb = FIAT.create_quadrature(interval, degree[1], scheme)
    q = FIAT.create_quadrature(extr_triangle, degree, scheme)
    assert len(q.get_points()) == len(qa.get_points())*len(qb.get_points())


@pytest.mark.parametrize(("points, degree"), tuple((p, d)
                                                   for p in range(2, 10)
                                                   for d in range(2*p - 2)))
def test_gauss_lobatto_legendre_quadrature(interval, points, degree):
    """Check that the quadrature rules correctly integrate all the right
    polynomial degrees."""

    q = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule(interval, points)

    assert numpy.round(q.integrate(lambda x: x[0]**degree) - 1./(degree+1), 14) == 0.


@pytest.mark.parametrize(("points, degree"), tuple((p, d)
                                                   for p in range(2, 10)
                                                   for d in range(2*p - 1)))
def test_radau_legendre_quadrature(interval, points, degree):
    """Check that the quadrature rules correctly integrate all the right
    polynomial degrees."""

    q = FIAT.quadrature.RadauQuadratureLineRule(interval, points)

    assert numpy.round(q.integrate(lambda x: x[0]**degree) - 1./(degree+1), 14) == 0.


@pytest.mark.parametrize(("points, degree"), tuple((p, d)
                                                   for p in range(2, 10)
                                                   for d in range(2*p)))
def test_gauss_legendre_quadrature(interval, points, degree):
    """Check that the quadrature rules correctly integrate all the right
    polynomial degrees."""

    q = FIAT.quadrature.GaussLegendreQuadratureLineRule(interval, points)

    assert numpy.round(q.integrate(lambda x: x[0]**degree) - 1./(degree+1), 14) == 0.


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
