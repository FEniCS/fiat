# Copyright (C) 2010 Marie E. Rognes
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
# Written by Marie E. Rognes <meg@simula.no> based on original
# implementation by Robert C. Kirby.
#
# Last changed: 2010-01-28

from . import finite_element, polynomial_set, dual_set , functional

def _initialize_entity_ids(topology):
    entity_ids = {}
    for (i, entity) in topology.items():
        entity_ids[i] = {}
        for j in entity:
            entity_ids[i][j] = []
    return entity_ids

class CrouzeixRaviartDualSet(dual_set.DualSet):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)."""

    def __init__(self, cell, degree):

        # Get topology dictionary
        d = cell.get_spatial_dimension()
        topology = cell.get_topology()

        # Initialize empty nodes and entity_ids
        entity_ids = _initialize_entity_ids(topology)
        nodes = [None for i in list(topology[d-1].keys())]

        # Construct nodes and entity_ids
        for i in topology[d-1]:

            # Construct midpoint
            x = cell.make_points(d-1, i, d)[0]

            # Degree of freedom number i is evaluation at midpoint
            nodes[i] = functional.PointEvaluation(cell, x)
            entity_ids[d-1][i] += [i]

        # Initialize super-class
        dual_set.DualSet.__init__(self, nodes, cell, entity_ids)

class CrouzeixRaviart(finite_element.FiniteElement):
    """The Crouzeix-Raviart finite element:

    K:                 Triangle/Tetrahedron
    Polynomial space:  P_1
    Dual basis:        Evaluation at facet midpoints
    """

    def __init__(self, cell, degree):

        # Crouzeix Raviart is only defined for polynomial degree == 1
        if not (degree == 1):
            raise Exception("Crouzeix-Raviart only defined for degree 1")

        # Construct polynomial spaces, dual basis and initialize
        # FiniteElement
        space = polynomial_set.ONPolynomialSet(cell, 1)
        dual = CrouzeixRaviartDualSet(cell, 1)
        finite_element.FiniteElement.__init__(self, space, dual, 1)


if __name__ == "__main__":

    from . import reference_element

    cells = [reference_element.UFCTriangle(),
             reference_element.UFCTetrahedron()]
    for cell in cells:
        print("Checking CrouzeixRaviart(cell, 1)")
        element = CrouzeixRaviart(cell, 1)
        print([L.pt_dict for L in element.dual_basis()])
        print()
