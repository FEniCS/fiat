__author__ = "Marie E. Rognes (meg@simula.no)"
__copyright__ = "Copyright (C) 2010 - Marie E. Rognes"
__license__  = "GNU LGPL version 3 or any later version"

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

      edges:    L(f) = f (x_i) * t       for (d+1) points x_i on each edge

      cell:     L(f) = \int f * g * dx   for g in RT_{k-1}

    Higher spatial dimensions are not yet implemented.
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

        # Extract spatial dimension
        d = cell.get_spatial_dimension()
        assert(d == 2), "Second kind Nedelecs only on triangles for now!"

        # Zero vertex-based degrees of freedom
        ids[0] = {}
        for vertex in range(3):
            ids[0][vertex] = []

        # (d+1) degrees of freedom per edge
        ids[1] = {}
        for edge in range(3):
            # Create points for evaluation of tangential components
            points = cell.make_points(d-1, edge, degree + 2)

            # A tangential component evaluation for each point
            dofs += [Tangent(cell, edge, point) for point in points]

            # Associate these dofs with this edge
            i = len(points)*edge
            ids[1][edge] = range(i, i+len(points))

        # Internal degrees of freedom
        ids[2] = {0: []}

        # Return if degree == 1 (no internals)
        if degree == 1: return (dofs, ids)

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

        # Associate these dofs with this edge
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
        assert(d == 2), "Second kind Nedelecs only implemented on triangles!"

        # Construct polynomial basis for d-vector fields
        Ps = ONPolynomialSet(cell, degree, (d, ))

        # Construct dual space
        Ls = NedelecSecondKindDual(cell, degree)

        # Set mapping
        mapping = "covariant piola"

        # Call init of super-class
        FiniteElement.__init__(self, Ps, Ls, degree, mapping=mapping)


if __name__=="__main__":

    from reference_element import UFCTriangle

    for k in range(1, 4):
        T = UFCTriangle()
        N2curl = NedelecSecondKind(T, k)

