# Copyright (C) 2010-2012 Marie E. Rognes
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy

from FIAT.finite_element import CiarletElement
from FIAT.dual_set import DualSet
from FIAT.polynomial_set import ONPolynomialSet
from FIAT.functional import PointEdgeTangentEvaluation as Tangent
from FIAT.functional import FrobeniusIntegralMoment as IntegralMoment
from FIAT.raviart_thomas import RaviartThomas
from FIAT.quadrature import make_quadrature, UFCTetrahedronFaceQuadratureRule
from FIAT.reference_element import UFCTetrahedron
from FIAT.check_format_variant import check_format_variant

from FIAT import polynomial_set, quadrature, functional


class NedelecSecondKindDual(DualSet):
    r"""
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

    def __init__(self, cell, degree, variant, quad_deg):

        # Define degrees of freedom
        (dofs, ids) = self.generate_degrees_of_freedom(cell, degree, variant, quad_deg)
        # Call init of super-class
        super(NedelecSecondKindDual, self).__init__(dofs, cell, ids)

    def generate_degrees_of_freedom(self, cell, degree, variant, quad_deg):
        "Generate dofs and geometry-to-dof maps (ids)."

        dofs = []
        ids = {}

        # Extract spatial dimension and topology
        d = cell.get_spatial_dimension()
        assert (d in (2, 3)), "Second kind Nedelecs only implemented in 2/3D."

        # Zero vertex-based degrees of freedom (d+1 of these)
        ids[0] = dict(list(zip(list(range(d + 1)), ([] for i in range(d + 1)))))

        # (d+1) degrees of freedom per entity of codimension 1 (edges)
        (edge_dofs, edge_ids) = self._generate_edge_dofs(cell, degree, 0, variant, quad_deg)
        dofs.extend(edge_dofs)
        ids[1] = edge_ids

        # Include face degrees of freedom if 3D
        if d == 3:
            (face_dofs, face_ids) = self._generate_face_dofs(cell, degree,
                                                             len(dofs), variant, quad_deg)
            dofs.extend(face_dofs)
            ids[2] = face_ids

        # Varying degrees of freedom (possibly zero) per cell
        (cell_dofs, cell_ids) = self._generate_cell_dofs(cell, degree, len(dofs), variant, quad_deg)
        dofs.extend(cell_dofs)
        ids[d] = cell_ids

        return (dofs, ids)

    def _generate_edge_dofs(self, cell, degree, offset, variant, quad_deg):
        """Generate degrees of freedoms (dofs) for entities of
        codimension 1 (edges)."""

        # (degree+1) tangential component point evaluation degrees of
        # freedom per entity of codimension 1 (edges)
        dofs = []
        ids = {}

        if variant == "integral":
            edge = cell.construct_subelement(1)
            Q = quadrature.make_quadrature(edge, quad_deg)
            Pq = polynomial_set.ONPolynomialSet(edge, degree)
            Pq_at_qpts = Pq.tabulate(Q.get_points())[tuple([0]*(1))]
            for e in range(len(cell.get_topology()[1])):
                for i in range(Pq_at_qpts.shape[0]):
                    phi = Pq_at_qpts[i, :]
                    dofs.append(functional.IntegralMomentOfEdgeTangentEvaluation(cell, Q, phi, e))
                jj = Pq_at_qpts.shape[0] * e
                ids[e] = list(range(offset + jj, offset + jj + Pq_at_qpts.shape[0]))

        elif variant == "point":
            for edge in range(len(cell.get_topology()[1])):

                # Create points for evaluation of tangential components
                points = cell.make_points(1, edge, degree + 2)

                # A tangential component evaluation for each point
                dofs += [Tangent(cell, edge, point) for point in points]

                # Associate these dofs with this edge
                i = len(points) * edge
                ids[edge] = list(range(offset + i, offset + i + len(points)))

        return (dofs, ids)

    def _generate_face_dofs(self, cell, degree, offset, variant, quad_deg):
        """Generate degrees of freedoms (dofs) for faces."""

        # Initialize empty dofs and identifiers (ids)
        dofs = []
        ids = dict(list(zip(list(range(4)), ([] for i in range(4)))))

        # Return empty info if not applicable
        d = cell.get_spatial_dimension()
        if (degree < 2):
            return (dofs, ids)

        msg = "2nd kind Nedelec face dofs only available with UFC convention"
        assert isinstance(cell, UFCTetrahedron), msg

        # Iterate over the faces of the tet
        num_faces = len(cell.get_topology()[2])
        for face in range(num_faces):

            # Construct quadrature scheme for this face
            m = 2 * (degree + 1)
            Q_face = UFCTetrahedronFaceQuadratureRule(face, m)

            # Construct Raviart-Thomas of (degree - 1) on the
            # reference face
            reference_face = Q_face.reference_rule().ref_el
            RT = RaviartThomas(reference_face, degree - 1, variant)
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
            scale = 1.0 / numpy.sqrt(numpy.linalg.det(numpy.dot(J.T, J)))
            phis = numpy.ndarray((d, num_quad_points))
            for i in range(num_rts):
                for q in range(num_quad_points):
                    phi_i_q = scale * numpy.dot(J, Phis[numpy.newaxis, i, :, q].T)
                    for j in range(d):
                        phis[j, q] = phi_i_q[j]

                # Construct degrees of freedom as integral moments on
                # this cell, using the special face quadrature
                # weighted against the values of the (physical)
                # Raviart--Thomas'es on the face
                dofs += [IntegralMoment(cell, Q_face, phis)]

            # Assign identifiers (num RTs per face + previous edge dofs)
            ids[face] = list(range(offset + num_rts*face, offset + num_rts*(face + 1)))

        return (dofs, ids)

    def _generate_cell_dofs(self, cell, degree, offset, variant, quad_deg):
        """Generate degrees of freedoms (dofs) for entities of
        codimension d (cells)."""

        # Return empty info if not applicable
        d = cell.get_spatial_dimension()
        if (d == 2 and degree < 2) or (d == 3 and degree < 3):
            return ([], {0: []})

        # Create quadrature points
        Q = make_quadrature(cell, 2 * (degree + 1))
        qs = Q.get_points()

        # Create Raviart-Thomas nodal basis
        RT = RaviartThomas(cell, degree + 1 - d, variant)
        phi = RT.get_nodal_basis()

        # Evaluate Raviart-Thomas basis at quadrature points
        phi_at_qs = phi.tabulate(qs)[(0,) * d]

        # Use (Frobenius) integral moments against RTs as dofs
        dofs = [IntegralMoment(cell, Q, phi_at_qs[i, :])
                for i in range(len(phi_at_qs))]

        # Associate these dofs with the interior
        ids = {0: list(range(offset, offset + len(dofs)))}
        return (dofs, ids)


class NedelecSecondKind(CiarletElement):
    """
    The H(curl) Nedelec elements of the second kind on triangles and
    tetrahedra: the polynomial space described by the full polynomials
    of degree k, with a suitable set of degrees of freedom to ensure
    H(curl) conformity.

    :arg ref_el: The reference element.
    :arg k: The degree.
    :arg variant: optional variant specifying the types of nodes.

    variant can be chosen from ["point", "integral", "integral(quadrature_degree)"]
    "point" -> dofs are evaluated by point evaluation. Note that this variant has suboptimal
    convergence order in the H(curl)-norm
    "integral" -> dofs are evaluated by quadrature rule. The quadrature degree is chosen to integrate
    polynomials of degree 5*k so that most expressions will be interpolated exactly. This is important
    when you want to have (nearly) curl-preserving interpolation.
    "integral(quadrature_degree)" -> dofs are evaluated by quadrature rule of degree quadrature_degree
    """

    def __init__(self, cell, k, variant=None):

        (variant, quad_deg) = check_format_variant(variant, k, "Nedelec Second Kind")

        # Check degree
        assert k >= 1, "Second kind Nedelecs start at 1!"

        # Get dimension
        d = cell.get_spatial_dimension()

        # Construct polynomial basis for d-vector fields
        Ps = ONPolynomialSet(cell, k, (d, ))

        # Construct dual space
        Ls = NedelecSecondKindDual(cell, k, variant, quad_deg)

        # Set form degree
        formdegree = 1  # 1-form

        # Set mapping
        mapping = "covariant piola"

        # Call init of super-class
        super(NedelecSecondKind, self).__init__(Ps, Ls, k, formdegree, mapping=mapping)
