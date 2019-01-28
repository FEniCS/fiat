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

from FIAT import finite_element, polynomial_set, dual_set, functional, P0, reference_element
from FIAT.discontinuous_lagrange import DiscontinuousLagrangeDualSet
from FIAT.reference_element import Point, DefaultLine, UFCInterval, UFCQuadrilateral, UFCHexahedron, UFCTriangle, UFCTetrahedron
from FIAT.P0 import P0Dual

hypercube_simplex_map = {'Point': Point(),
                         'DefaultLine': DefaultLine(),
                         'UFCInterval': UFCInterval(),
                         'UFCQuadrilateral': UFCTriangle(),
                         'UFCHexahedron': UFCTetrahedron()}

class DPC0Dual(P0Dual):
    def __init__(self, ref_el):
        super(DPC0Dual, self).__init__(ref_el)

class DPC0(finite_element.CiarletElement):
    def __init__(self, ref_el):
        poly_set = polynomial_set.ONPolynomialSet(hypercube_simplex_map[type(ref_el).__name__], 0)
        dual = DPC0Dual(ref_el)
        degree = 0
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(DPC0, self).__init__(poly_set=poly_set,
                                   dual=dual,
                                   order=degree,
                                   ref_el=ref_el,
                                   formdegree=formdegree)


class DiscontinuousSerendipityDualSet(DiscontinuousLagrangeDualSet):
    """The dual basis for Serendipity elements.  This class works for
    hypercubes of any dimension.  Nodes are point evaluation at
    equispaced points.  This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""

    def __init__(self, ref_el, degree):
        super(DiscontinuousSerendipityDualSet, self).__init__(ref_el, degree)


class HigherOrderDiscontinuousSerendipity(finite_element.CiarletElement):
    """The discontinuous Serendipity finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        poly_set = polynomial_set.ONPolynomialSet(hypercube_simplex_map[type(ref_el).__name__], degree)
        dual = DiscontinuousSerendipityDualSet(hypercube_simplex_map[type(ref_el).__name__], degree)
        formdegree = ref_el.get_spatial_dimension()  # n-form
        super(HigherOrderDiscontinuousSerendipity, self).__init__(poly_set=poly_set,
                                                                  dual=dual,
                                                                  order=degree,
                                                                  ref_el=ref_el,
                                                                  formdegree=formdegree)


def DiscontinuousSerendipity(ref_el, degree):
    if degree == 0:
        return DPC0(ref_el)
    else:
        return HigherOrderDiscontinuousSerendipity(ref_el, degree)
