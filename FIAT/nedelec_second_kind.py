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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT.  If not, see <http://www.gnu.org/licenses/>.

from finite_element import FiniteElement
from dual_set import DualSet
from polynomial_set import ONPolynomialSet
from functional import PointEdgeTangentEvaluation as Tangent
from functional import FrobeniusIntegralMoment as IntegralMoment
from raviart_thomas import RaviartThomas
from quadrature import make_quadrature

class NedelecSecondKindDual(DualSet):
    """
    This class represents the dual basis for the Nedelec H(curl)
    elements of the second kind. The degrees of freedom (L) for the
    elements of the k'th degree are

    d = 2:

      vertices: None

      edges:    L(f) = f (x_i) * t       for (k+1) points x_i on each edge

      cell:     L(f) = \int f * g * dx   for g in RT_{k-1}


   d = 3:

      vertices: None

      edges:    L(f)  = f(x_i) * t       for (k+1) points x_i on each edge

      faces:

      cell:

    Higher spatial dimensions are not yet implemented. (For d = 1,
    these elements coincide with the CG_k elements.)
    """

    def __init__ (self, cell, degree):

        # Define degrees of freedom
        (dofs, ids) = self.generate_degrees_of_freedom(cell, degree)

        # Call init of super-class
        DualSet.__init__(self, dofs, cell, ids)

    def generate_degrees_of_freedom(self, cell, degree):
        "Generate dofs and geometry-to-dof maps (ids)."

        dofs = []
        ids = {}

        # Extract spatial dimension and topology
        d = cell.get_spatial_dimension()
        topology = cell.get_topology()

        # Zero vertex-based degrees of freedom (d+1 of these)
        ids[0] = dict(zip(range(d+1), ([] for i in range(d+1))))

        # (d+1) degrees of freedom per entity of codimension 1 (edges)
        ids[1] = {}
        for edge in range(len(topology[1])):

            # Create points for evaluation of tangential components
            points = cell.make_points(1, edge, degree + 2)

            # A tangential component evaluation for each point
            dofs += [Tangent(cell, edge, point) for point in points]

            # Associate these dofs with this edge
            i = len(points)*edge
            ids[1][edge] = range(i, i+len(points))


        # If this is lowest order element, we just need to fill up ids
        # with appropriate amounts of empty lists
        ids[d] = {0: []}
        if degree == 1:
            if d == 2:
                return (dofs, ids)

            ids[2] = dict(zip(range(4), ([] for i in range(4))))
            return (dofs, ids)

        assert(d == 2), "Only lowest order 2nd kind Nedelecs implemented on tetrahedra"

        # Create quadrature points
        Q = make_quadrature(cell, 2*(degree+1))
        qs = Q.get_points()

        # Create Raviart-Thomas nodal basis
        RT = RaviartThomas(cell, degree - 1)
        phi = RT.get_nodal_basis()

        # Evaluate Raviart-Thomas basis at quadrature points
        phi_at_qs = phi.tabulate(qs)[(0, 0)]

        # Use (Frobenius) integral moments against RTs as dofs
        dofs += [IntegralMoment(cell, Q, phi_at_qs[i, :])
                 for i in range(len(phi_at_qs))]

        # Associate these dofs with the interior
        i = 3*(degree+1)
        ids[2][0] = range(i, i+len(phi_at_qs))

        return (dofs, ids)


class NedelecSecondKind(FiniteElement):
    """

    """

    def __init__(self, cell, degree):

        # Check degree
        assert(degree >= 1), "Second kind Nedelecs start at 1!"

        # Get dimension
        d = cell.get_spatial_dimension()

        # Construct polynomial basis for d-vector fields
        Ps = ONPolynomialSet(cell, degree, (d, ))

        # Construct dual space
        Ls = NedelecSecondKindDual(cell, degree)

        # Set mapping
        mapping = "covariant piola"

        # Call init of super-class
        FiniteElement.__init__(self, Ps, Ls, degree, mapping=mapping)


if __name__=="__main__":

    from reference_element import UFCTriangle, UFCTetrahedron

    for k in range(1, 4):
        T = UFCTriangle()
        N2curl = NedelecSecondKind(T, k)

    for k in range(1, 2):
        T = UFCTetrahedron()
        N2curl = NedelecSecondKind(T, k)

