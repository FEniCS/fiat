# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import (polynomial_set, expansions, quadrature, dual_set,
                  finite_element, functional)
from itertools import chain
import numpy
from FIAT.check_format_variant import check_format_variant


def NedelecSpace2D(ref_el, k):
    """Constructs a basis for the 2d H(curl) space of the first kind
    which is (P_k)^2 + P_k rot( x )"""
    sd = ref_el.get_spatial_dimension()
    if sd != 2:
        raise Exception("NedelecSpace2D requires 2d reference element")

    vec_Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1, (sd,))

    dimPkp1 = expansions.polynomial_dimension(ref_el, k + 1)
    dimPk = expansions.polynomial_dimension(ref_el, k)
    dimPkm1 = expansions.polynomial_dimension(ref_el, k - 1)

    vec_Pk_indices = list(chain(*(range(i * dimPkp1, i * dimPkp1 + dimPk)
                                  for i in range(sd))))
    vec_Pk_from_Pkp1 = vec_Pkp1.take(vec_Pk_indices)

    Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1)
    PkH = Pkp1.take(list(range(dimPkm1, dimPk)))

    Q = quadrature.make_quadrature(ref_el, 2 * k + 2)

    Qpts = numpy.array(Q.get_points())
    Qwts = numpy.array(Q.get_weights())

    zero_index = tuple([0 for i in range(sd)])

    PkH_at_Qpts = PkH.tabulate(Qpts)[zero_index]
    Pkp1_at_Qpts = Pkp1.tabulate(Qpts)[zero_index]

    PkH_crossx_coeffs = numpy.zeros((PkH.get_num_members(),
                                     sd,
                                     Pkp1.get_num_members()), "d")

    def rot_x_foo(a):
        if a == 0:
            return 1, 1.0
        elif a == 1:
            return 0, -1.0

    for i in range(PkH.get_num_members()):
        for j in range(sd):
            (ind, sign) = rot_x_foo(j)
            for k in range(Pkp1.get_num_members()):
                PkH_crossx_coeffs[i, j, k] = sign * sum(Qwts * PkH_at_Qpts[i, :] * Qpts[:, ind] * Pkp1_at_Qpts[k, :])
#                for l in range( len( Qpts ) ):
#                    PkH_crossx_coeffs[i,j,k] += Qwts[ l ] \
#                                                * PkH_at_Qpts[i,l] \
#                                                * Qpts[l][ind] \
#                                                * Pkp1_at_Qpts[k,l] \
#                                                * sign

    PkHcrossx = polynomial_set.PolynomialSet(ref_el,
                                             k + 1,
                                             k + 1,
                                             vec_Pkp1.get_expansion_set(),
                                             PkH_crossx_coeffs,
                                             vec_Pkp1.get_dmats())

    return polynomial_set.polynomial_set_union_normalized(vec_Pk_from_Pkp1,
                                                          PkHcrossx)


def NedelecSpace3D(ref_el, k):
    """Constructs a nodal basis for the 3d first-kind Nedelec space"""
    sd = ref_el.get_spatial_dimension()
    if sd != 3:
        raise Exception("NedelecSpace3D requires 3d reference element")

    vec_Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1, (sd,))

    dimPkp1 = expansions.polynomial_dimension(ref_el, k + 1)
    dimPk = expansions.polynomial_dimension(ref_el, k)
    if k > 0:
        dimPkm1 = expansions.polynomial_dimension(ref_el, k - 1)
    else:
        dimPkm1 = 0

    vec_Pk_indices = list(chain(*(range(i * dimPkp1, i * dimPkp1 + dimPk)
                                  for i in range(sd))))
    vec_Pk = vec_Pkp1.take(vec_Pk_indices)

    vec_Pke_indices = list(chain(*(range(i * dimPkp1 + dimPkm1, i * dimPkp1 + dimPk)
                                   for i in range(sd))))

    vec_Pke = vec_Pkp1.take(vec_Pke_indices)

    Pkp1 = polynomial_set.ONPolynomialSet(ref_el, k + 1)

    Q = quadrature.make_quadrature(ref_el, 2 * (k + 1))

    Qpts = numpy.array(Q.get_points())
    Qwts = numpy.array(Q.get_weights())

    zero_index = tuple([0 for i in range(sd)])

    PkCrossXcoeffs = numpy.zeros((vec_Pke.get_num_members(),
                                  sd,
                                  Pkp1.get_num_members()), "d")

    Pke_qpts = vec_Pke.tabulate(Qpts)[zero_index]
    Pkp1_at_Qpts = Pkp1.tabulate(Qpts)[zero_index]

    for i in range(vec_Pke.get_num_members()):
        for j in range(sd):  # vector components
            qwts_cur_bf_val = (
                Qpts[:, (j + 2) % 3] * Pke_qpts[i, (j + 1) % 3, :] -
                Qpts[:, (j + 1) % 3] * Pke_qpts[i, (j + 2) % 3, :]) * Qwts
            PkCrossXcoeffs[i, j, :] = numpy.dot(Pkp1_at_Qpts, qwts_cur_bf_val)
#            for k in range( Pkp1.get_num_members() ):
#                 PkCrossXcoeffs[i,j,k] = sum( Qwts * cur_bf_val * Pkp1_at_Qpts[k,:] )
#                for l in range( len( Qpts ) ):
#                    cur_bf_val = Qpts[l][(j+2)%3] \
#                                 * Pke_qpts[i,(j+1)%3,l] \
#                                 - Qpts[l][(j+1)%3] \
#                                 * Pke_qpts[i,(j+2)%3,l]
#                    PkCrossXcoeffs[i,j,k] += Qwts[l] \
#                                             * cur_bf_val \
#                                             * Pkp1_at_Qpts[k,l]

    PkCrossX = polynomial_set.PolynomialSet(ref_el,
                                            k + 1,
                                            k + 1,
                                            vec_Pkp1.get_expansion_set(),
                                            PkCrossXcoeffs,
                                            vec_Pkp1.get_dmats())
    return polynomial_set.polynomial_set_union_normalized(vec_Pk, PkCrossX)


class NedelecDual2D(dual_set.DualSet):
    """Dual basis for first-kind Nedelec in 2D."""

    def __init__(self, ref_el, degree, variant, quad_deg):
        sd = ref_el.get_spatial_dimension()
        if sd != 2:
            raise Exception("Nedelec2D only works on triangles")

        nodes = []

        t = ref_el.get_topology()

        if variant == "integral":
            # edge nodes are \int_F v\cdot t p ds where p \in P_{q-1}(edge)
            # degree is q - 1
            edge = ref_el.get_facet_element()
            Q = quadrature.make_quadrature(edge, quad_deg)
            Pq = polynomial_set.ONPolynomialSet(edge, degree)
            Pq_at_qpts = Pq.tabulate(Q.get_points())[tuple([0]*(sd - 1))]
            for e in range(len(t[sd - 1])):
                for i in range(Pq_at_qpts.shape[0]):
                    phi = Pq_at_qpts[i, :]
                    nodes.append(functional.IntegralMomentOfEdgeTangentEvaluation(ref_el, Q, phi, e))

            # internal nodes. These are \int_T v \cdot p dx where p \in P_{q-2}^2
            if degree > 0:
                Q = quadrature.make_quadrature(ref_el, quad_deg)
                qpts = Q.get_points()
                Pkm1 = polynomial_set.ONPolynomialSet(ref_el, degree - 1)
                zero_index = tuple([0 for i in range(sd)])
                Pkm1_at_qpts = Pkm1.tabulate(qpts)[zero_index]

                for d in range(sd):
                    for i in range(Pkm1_at_qpts.shape[0]):
                        phi_cur = Pkm1_at_qpts[i, :]
                        l_cur = functional.IntegralMoment(ref_el, Q, phi_cur, (d,), (sd,))
                        nodes.append(l_cur)

        elif variant == "point":
            num_edges = len(t[1])

            # edge tangents
            for i in range(num_edges):
                pts_cur = ref_el.make_points(1, i, degree + 2)
                for j in range(len(pts_cur)):
                    pt_cur = pts_cur[j]
                    f = functional.PointEdgeTangentEvaluation(ref_el, i, pt_cur)
                    nodes.append(f)

            # internal moments
            if degree > 0:
                Q = quadrature.make_quadrature(ref_el, 2 * (degree + 1))
                qpts = Q.get_points()
                Pkm1 = polynomial_set.ONPolynomialSet(ref_el, degree - 1)
                zero_index = tuple([0 for i in range(sd)])
                Pkm1_at_qpts = Pkm1.tabulate(qpts)[zero_index]

                for d in range(sd):
                    for i in range(Pkm1_at_qpts.shape[0]):
                        phi_cur = Pkm1_at_qpts[i, :]
                        l_cur = functional.IntegralMoment(ref_el, Q, phi_cur, (d,), (sd,))
                        nodes.append(l_cur)

        entity_ids = {}

        # set to empty
        for i in range(sd + 1):
            entity_ids[i] = {}
            for j in range(len(t[i])):
                entity_ids[i][j] = []

        cur = 0

        # edges
        num_edge_pts = len(ref_el.make_points(1, 0, degree + 2))

        for i in range(len(t[1])):
            entity_ids[1][i] = list(range(cur, cur + num_edge_pts))
            cur += num_edge_pts

        # moments against P_{degree-1} internally, if degree > 0
        if degree > 0:
            num_internal_dof = sd * Pkm1_at_qpts.shape[0]
            entity_ids[2][0] = list(range(cur, cur + num_internal_dof))

        super(NedelecDual2D, self).__init__(nodes, ref_el, entity_ids)


class NedelecDual3D(dual_set.DualSet):
    """Dual basis for first-kind Nedelec in 3D."""

    def __init__(self, ref_el, degree, variant, quad_deg):
        sd = ref_el.get_spatial_dimension()
        if sd != 3:
            raise Exception("NedelecDual3D only works on tetrahedra")

        nodes = []

        t = ref_el.get_topology()

        if variant == "integral":
            # edge nodes are \int_F v\cdot t p ds where p \in P_{q-1}(edge)
            # degree is q - 1
            edge = ref_el.get_facet_element().get_facet_element()
            Q = quadrature.make_quadrature(edge, quad_deg)
            Pq = polynomial_set.ONPolynomialSet(edge, degree)
            Pq_at_qpts = Pq.tabulate(Q.get_points())[tuple([0]*(1))]
            for e in range(len(t[1])):
                for i in range(Pq_at_qpts.shape[0]):
                    phi = Pq_at_qpts[i, :]
                    nodes.append(functional.IntegralMomentOfEdgeTangentEvaluation(ref_el, Q, phi, e))

            # face nodes are \int_F v\cdot p dA where p \in P_{q-2}(f)^3 with p \cdot n = 0 (cmp. Monk)
            # these are equivalent to dofs from Fenics book defined by
            # \int_F v\times n \cdot p ds where p \in P_{q-2}(f)^2
            if degree > 0:
                facet = ref_el.get_facet_element()
                Q = quadrature.make_quadrature(facet, quad_deg)
                Pq = polynomial_set.ONPolynomialSet(facet, degree-1, (sd,))
                Pq_at_qpts = Pq.tabulate(Q.get_points())[(0, 0)]

                for f in range(len(t[2])):
                    # R is used to map [1,0,0] to tangent1 and [0,1,0] to tangent2
                    R = ref_el.compute_face_tangents(f)

                    # Skip last functionals because we only want p with p \cdot n = 0
                    for i in range(2 * Pq.get_num_members() // 3):
                        phi = Pq_at_qpts[i, ...]
                        phi = numpy.matmul(phi[:-1, ...].T, R)
                        nodes.append(functional.MonkIntegralMoment(ref_el, Q, phi, f))

            # internal nodes. These are \int_T v \cdot p dx where p \in P_{q-3}^3(T)
            if degree > 1:
                Q = quadrature.make_quadrature(ref_el, quad_deg)
                qpts = Q.get_points()
                Pkm2 = polynomial_set.ONPolynomialSet(ref_el, degree - 2)
                zero_index = tuple([0 for i in range(sd)])
                Pkm2_at_qpts = Pkm2.tabulate(qpts)[zero_index]

                for d in range(sd):
                    for i in range(Pkm2_at_qpts.shape[0]):
                        phi_cur = Pkm2_at_qpts[i, :]
                        l_cur = functional.IntegralMoment(ref_el, Q, phi_cur, (d,), (sd,))
                        nodes.append(l_cur)

        elif variant == "point":
            num_edges = len(t[1])

            for i in range(num_edges):
                # points to specify P_k on each edge
                pts_cur = ref_el.make_points(1, i, degree + 2)
                for j in range(len(pts_cur)):
                    pt_cur = pts_cur[j]
                    f = functional.PointEdgeTangentEvaluation(ref_el, i, pt_cur)
                    nodes.append(f)

            if degree > 0:  # face tangents
                num_faces = len(t[2])
                for i in range(num_faces):  # loop over faces
                    pts_cur = ref_el.make_points(2, i, degree + 2)
                    for j in range(len(pts_cur)):  # loop over points
                        pt_cur = pts_cur[j]
                        for k in range(2):  # loop over tangents
                            f = functional.PointFaceTangentEvaluation(ref_el, i, k, pt_cur)
                            nodes.append(f)

            if degree > 1:  # internal moments
                Q = quadrature.make_quadrature(ref_el, 2 * (degree + 1))
                qpts = Q.get_points()
                Pkm2 = polynomial_set.ONPolynomialSet(ref_el, degree - 2)
                zero_index = tuple([0 for i in range(sd)])
                Pkm2_at_qpts = Pkm2.tabulate(qpts)[zero_index]

                for d in range(sd):
                    for i in range(Pkm2_at_qpts.shape[0]):
                        phi_cur = Pkm2_at_qpts[i, :]
                        f = functional.IntegralMoment(ref_el, Q, phi_cur, (d,), (sd,))
                        nodes.append(f)

        entity_ids = {}
        # set to empty
        for i in range(sd + 1):
            entity_ids[i] = {}
            for j in range(len(t[i])):
                entity_ids[i][j] = []

        cur = 0

        # edge dof
        num_pts_per_edge = len(ref_el.make_points(1, 0, degree + 2))
        for i in range(len(t[1])):
            entity_ids[1][i] = list(range(cur, cur + num_pts_per_edge))
            cur += num_pts_per_edge

        # face dof
        if degree > 0:
            num_pts_per_face = len(ref_el.make_points(2, 0, degree + 2))
            for i in range(len(t[2])):
                entity_ids[2][i] = list(range(cur, cur + 2 * num_pts_per_face))
                cur += 2 * num_pts_per_face

        if degree > 1:
            num_internal_dof = Pkm2_at_qpts.shape[0] * sd
            entity_ids[3][0] = list(range(cur, cur + num_internal_dof))

        super(NedelecDual3D, self).__init__(nodes, ref_el, entity_ids)


class Nedelec(finite_element.CiarletElement):
    """
    Nedelec finite element

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

    def __init__(self, ref_el, k, variant=None):

        degree = k - 1

        (variant, quad_deg) = check_format_variant(variant, degree, "Nedelec")

        if ref_el.get_spatial_dimension() == 3:
            poly_set = NedelecSpace3D(ref_el, degree)
            dual = NedelecDual3D(ref_el, degree, variant, quad_deg)
        elif ref_el.get_spatial_dimension() == 2:
            poly_set = NedelecSpace2D(ref_el, degree)
            dual = NedelecDual2D(ref_el, degree, variant, quad_deg)
        else:
            raise Exception("Not implemented")
        formdegree = 1  # 1-form
        super(Nedelec, self).__init__(poly_set, dual, degree, formdegree,
                                      mapping="covariant piola")
