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

from FIAT import finite_element, polynomial_set, dual_set, functional, P0,
                 reference_element
from FIAT.discontinuous_lagrange import DiscontinuousLagrangeDualSet


class DiscontinuousLagrangeCubeDualSet(DiscontinuousLagrangeDualSet):
    """The dual basis for Lagrange elements.  This class works for
    hypercubes of any dimension.  Nodes are point evaluation at
    equispaced points.  This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""

    def __init__(self, ref_el, degree):
        super(DiscontinuousLagrangeCubeDualSet, self).__init__(ref_el, degree)


class HigherOrderDiscontinuousLagrangeCube(finite_element.CiarletElement):
    """The discontinuous Lagrange finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        hypercube_simplex_map = {Point: Point,
                                 DefaultLine: DefaultLine,
                                 UFCInterval: UFCInterval,
                                 UFCQuadrilateral: UFCTriangle,
                                 UFCHexahedron: UFCTetrahedron}
        poly_set = polynomial_set.ONPolynomialSet(hypercube_simplex_map[ref_el], degree)
        dual = DiscontinuousLagrangeCubeDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(HigherOrderDiscontinuousLagrangeCube, self).__init__(poly_set, dual, degree, formdegree)


def DiscontinuousLagrangeCube(ref_el, degree):
    if degree == 0:
        hypercube_simplex_map = {Point: Point,
                                 DefaultLine: DefaultLine,
                                 UFCInterval: UFCInterval,
                                 UFCQuadrilateral: UFCTriangle,
                                 UFCHexahedron: UFCTetrahedron}
        return P0.P0(hypercube_simplex_map[ref_el])
    else:
        return HigherOrderDiscontinuousLagrangeCube(ref_el, degree)
