# Copyright (C) 2010-2012 Marie E. Rognes
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

import numpy

from .finite_element import FiniteElement
from .dual_set import DualSet
from .polynomial_set import ONPolynomialSet
from .functional import PointEdgeTangentEvaluation as Tangent
from .functional import FrobeniusIntegralMoment as IntegralMoment
from .raviart_thomas import RaviartThomas
from .quadrature import make_quadrature, UFCTetrahedronFaceQuadratureRule
from .reference_element import UFCTriangle, UFCTetrahedron

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

      edges:    L(f)  = f(x_i) * t         for (k+1) points x_i on each edge

      faces:    L(f) = \int_F f * g * ds   for g in RT_{k-1}(F) for each face F

      cell:     L(f) = \int f * g * dx     for g in RT_{k-2}

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
        assert (d in (2, 3)), "Second kind Nedelecs only implemented in 2/3D."
        topology = cell.get_topology()

        # Zero vertex-based degrees of freedom (d+1 of these)
        ids[0] = dict(list(zip(list(range(d+1)), ([] for i in range(d+1)))))

        # (d+1) degrees of freedom per entity of codimension 1 (edges)
        (edge_dofs, edge_ids) = self._generate_edge_dofs(cell, degree, 0)
        dofs.extend(edge_dofs)
        ids[1] = edge_ids

        # Include face degrees of freedom if 3D
        if d == 3:
            (face_dofs, face_ids) = self._generate_face_dofs(cell, degree,
                                                             len(dofs))
            dofs.extend(face_dofs)
            ids[2] = face_ids

        # Varying degrees of freedom (possibly zero) per cell
        (cell_dofs, cell_ids) = self._generate_cell_dofs(cell, degree,len(dofs))
        dofs.extend(cell_dofs)
        ids[d] = cell_ids

        return (dofs, ids)

    def _generate_edge_dofs(self, cell, degree, offset):
        """Generate degrees of freedoms (dofs) for entities of
        codimension 1 (edges)."""

        # (degree+1) tangential component point evaluation degrees of
        # freedom per entity of codimension 1 (edges)
        dofs = []
        ids = {}
        for edge in range(len(cell.get_topology()[1])):

            # Create points for evaluation of tangential components
            points = cell.make_points(1, edge, degree + 2)

            # A tangential component evaluation for each point
            dofs += [Tangent(cell, edge, point) for point in points]

            # Associate these dofs with this edge
            i = len(points)*edge
            ids[edge] = list(range(offset + i, offset + i + len(points)))

        return (dofs, ids)

    def _generate_face_dofs(self, cell, degree, offset):
        """Generate degrees of freedoms (dofs) for faces."""

        # Initialize empty dofs and identifiers (ids)
        dofs = []
        ids = dict(list(zip(list(range(4)), ([] for i in range(4)))))

        # Return empty info if not applicable
        d = cell.get_spatial_dimension()
        if (degree < 2):
            return (dofs, ids)

        msg = "2nd kind Nedelec face dofs only available with UFC convention"
        assert isinstance(cell, UFCTetrahedron),  msg

        # Iterate over the faces of the tet
        num_faces = len(cell.get_topology()[2])
        for face in range(num_faces):

            # Construct quadrature scheme for this face
            m = 2*(degree + 1)
            Q_face = UFCTetrahedronFaceQuadratureRule(face, m)
            quad_points = Q_face.get_points()

            # Construct Raviart-Thomas of (degree - 1) on the
            # reference face
            reference_face = Q_face.reference_rule().ref_el
            RT = RaviartThomas(reference_face, degree - 1)
            num_rts = RT.space_dimension()

            # Evaluate RT basis functions at reference quadrature
            # points
            ref_quad_points = Q_face.reference_rule().get_points()
            num_quad_points = len(ref_quad_points)
            Phi = RT.get_nodal_basis()
            Phis = Phi.tabulate(ref_quad_points)[(0, 0)]

            # Note: Phis has dimensions:
            # num_basis_functions x num_components x num_quad_points

            # Map Phis -> phis (reference values to physical values)
            J = Q_face.jacobian()
            scale = 1.0/numpy.sqrt(numpy.linalg.det(J.transpose()*J))
            phis = numpy.ndarray((d, num_quad_points))
            for i in range(num_rts):
                for q in range(num_quad_points):
                    phi_i_q = scale*J*numpy.matrix(Phis[i, :, q]).transpose()
                    for j in range(d):
                        phis[j, q] = phi_i_q[j]

                # Construct degrees of freedom as integral moments on
                # this cell, using the special face quadrature
                # weighted against the values of the (physical)
                # Raviart--Thomas'es on the face
                dofs += [IntegralMoment(cell, Q_face, phis)]

            # Assign identifiers (num RTs per face + previous edge dofs)
            ids[face] = list(range(offset + num_rts*face, offset + num_rts*(face+1)))

        return (dofs, ids)

    def _generate_cell_dofs(self, cell, degree, offset):
        """Generate degrees of freedoms (dofs) for entities of
        codimension d (cells)."""

        # Return empty info if not applicable
        d = cell.get_spatial_dimension()
        if (d == 2 and degree < 2) or (d == 3 and degree < 3):
            return ([], {0: []})

        # Create quadrature points
        Q = make_quadrature(cell, 2*(degree+1))
        qs = Q.get_points()

        # Create Raviart-Thomas nodal basis
        RT = RaviartThomas(cell, degree + 1 - d)
        phi = RT.get_nodal_basis()

        # Evaluate Raviart-Thomas basis at quadrature points
        phi_at_qs = phi.tabulate(qs)[(0,)*d]

        # Use (Frobenius) integral moments against RTs as dofs
        dofs = [IntegralMoment(cell, Q, phi_at_qs[i, :])
                for i in range(len(phi_at_qs))]

        # Associate these dofs with the interior
        ids = {0: list(range(offset, offset + len(dofs)))}
        return (dofs, ids)

class NedelecSecondKind(FiniteElement):
    """
    The H(curl) Nedelec elements of the second kind on triangles and
    tetrahedra: the polynomial space described by the full polynomials
    of degree k, with a suitable set of degrees of freedom to ensure
    H(curl) conformity.
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

    for k in range(1, 4):
        T = UFCTriangle()
        N2curl = NedelecSecondKind(T, k)

    for k in range(1, 4):
        T = UFCTetrahedron()
        N2curl = NedelecSecondKind(T, k)
        Nfs = N2curl.get_nodal_basis()
        pts = T.make_lattice( 1 )
        vals = Nfs.tabulate( pts , 1 )

