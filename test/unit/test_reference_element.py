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

from __future__ import absolute_import, print_function, division

import pytest
import numpy as np

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import Point, FiredrakeQuadrilateral, TensorProductCell

point = Point()
interval = UFCInterval()
triangle = UFCTriangle()
quadrilateral = FiredrakeQuadrilateral()
tetrahedron = UFCTetrahedron()
interval_x_interval = TensorProductCell(interval, interval)
triangle_x_interval = TensorProductCell(triangle, interval)
quadrilateral_x_interval = TensorProductCell(quadrilateral, interval)


@pytest.mark.parametrize(('cell', 'volume'),
                         [pytest.mark.xfail((point, 1)),
                          (interval, 1),
                          (triangle, 1/2),
                          (quadrilateral, 1),
                          (tetrahedron, 1/6),
                          (interval_x_interval, 1),
                          (triangle_x_interval, 1/2),
                          (quadrilateral_x_interval, 1)])
def test_volume(cell, volume):
    assert np.allclose(volume, cell.volume())


@pytest.mark.parametrize(('cell', 'normals'),
                         [(interval, [[-1],
                                      [1]]),
                          (triangle, [[1, 1],
                                      [-1, 0],
                                      [0, -1]]),
                          (quadrilateral, [[-1, 0],
                                           [1, 0],
                                           [0, -1],
                                           [0, 1]]),
                          (tetrahedron, [[1, 1, 1],
                                         [-1, 0, 0],
                                         [0, -1, 0],
                                         [0, 0, -1]])])
def test_reference_normal(cell, normals):
    facet_dim = cell.get_spatial_dimension() - 1
    for facet_number in range(len(cell.get_topology()[facet_dim])):
        assert np.allclose(normals[facet_number],
                           cell.compute_reference_normal(facet_dim, facet_number))


@pytest.mark.parametrize('cell',
                         [interval_x_interval,
                          triangle_x_interval,
                          quadrilateral_x_interval])
def test_reference_normal_horiz(cell):
    dim = cell.get_spatial_dimension()
    np.allclose((0,) * (dim - 1) + (-1,),
                cell.compute_reference_normal((dim - 1, 0), 0))  # bottom facet
    np.allclose((0,) * (dim - 1) + (1,),
                cell.compute_reference_normal((dim - 1, 0), 1))  # top facet


@pytest.mark.parametrize(('cell', 'normals'),
                         [(interval_x_interval, [[-1, 0],
                                                 [1, 0]]),
                          (triangle_x_interval, [[1, 1, 0],
                                                 [-1, 0, 0],
                                                 [0, -1, 0]]),
                          (quadrilateral_x_interval, [[-1, 0, 0],
                                                      [1, 0, 0],
                                                      [0, -1, 0],
                                                      [0, 1, 0]])])
def test_reference_normal_vert(cell, normals):
    dim = cell.get_spatial_dimension()
    vert_dim = (dim - 2, 1)
    for facet_number in range(len(cell.get_topology()[vert_dim])):
        assert np.allclose(normals[facet_number],
                           cell.compute_reference_normal(vert_dim, facet_number))
        
        
if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
