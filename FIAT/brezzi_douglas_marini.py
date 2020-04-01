# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import (finite_element, quadrature, functional, dual_set,
                  polynomial_set, nedelec)
from FIAT.check_format_variant import check_format_variant


class BDMDualSet(dual_set.DualSet):
    def __init__(self, ref_el, degree, variant, quad_deg):

        # Initialize containers for map: mesh_entity -> dof number and
        # dual basis
        entity_ids = {}
        nodes = []

        sd = ref_el.get_spatial_dimension()
        t = ref_el.get_topology()

        if variant == "integral":
            facet = ref_el.get_facet_element()
            # Facet nodes are \int_F v\cdot n p ds where p \in P_{q-1}
            # degree is q - 1
            Q = quadrature.make_quadrature(facet, quad_deg)
            Pq = polynomial_set.ONPolynomialSet(facet, degree)
            Pq_at_qpts = Pq.tabulate(Q.get_points())[tuple([0]*(sd - 1))]
            for f in range(len(t[sd - 1])):
                for i in range(Pq_at_qpts.shape[0]):
                    phi = Pq_at_qpts[i, :]
                    nodes.append(functional.IntegralMomentOfScaledNormalEvaluation(ref_el, Q, phi, f))

            # internal nodes
            if degree > 1:
                Q = quadrature.make_quadrature(ref_el, quad_deg)
                qpts = Q.get_points()
                Nedel = nedelec.Nedelec(ref_el, degree - 1, variant)
                Nedfs = Nedel.get_nodal_basis()
                zero_index = tuple([0 for i in range(sd)])
                Ned_at_qpts = Nedfs.tabulate(qpts)[zero_index]

                for i in range(len(Ned_at_qpts)):
                    phi_cur = Ned_at_qpts[i, :]
                    l_cur = functional.FrobeniusIntegralMoment(ref_el, Q, phi_cur)
                    nodes.append(l_cur)

        elif variant == "point":
            # Define each functional for the dual set
            # codimension 1 facets
            for i in range(len(t[sd - 1])):
                pts_cur = ref_el.make_points(sd - 1, i, sd + degree)
                for j in range(len(pts_cur)):
                    pt_cur = pts_cur[j]
                    f = functional.PointScaledNormalEvaluation(ref_el, i, pt_cur)
                    nodes.append(f)

            # internal nodes
            if degree > 1:
                Q = quadrature.make_quadrature(ref_el, 2 * (degree + 1))
                qpts = Q.get_points()
                Nedel = nedelec.Nedelec(ref_el, degree - 1, variant)
                Nedfs = Nedel.get_nodal_basis()
                zero_index = tuple([0 for i in range(sd)])
                Ned_at_qpts = Nedfs.tabulate(qpts)[zero_index]

                for i in range(len(Ned_at_qpts)):
                    phi_cur = Ned_at_qpts[i, :]
                    l_cur = functional.FrobeniusIntegralMoment(ref_el, Q, phi_cur)
                    nodes.append(l_cur)

        # sets vertices (and in 3d, edges) to have no nodes
        for i in range(sd - 1):
            entity_ids[i] = {}
            for j in range(len(t[i])):
                entity_ids[i][j] = []

        cur = 0

        # set codimension 1 (edges 2d, faces 3d) dof
        pts_facet_0 = ref_el.make_points(sd - 1, 0, sd + degree)
        pts_per_facet = len(pts_facet_0)

        entity_ids[sd - 1] = {}
        for i in range(len(t[sd - 1])):
            entity_ids[sd - 1][i] = list(range(cur, cur + pts_per_facet))
            cur += pts_per_facet

        # internal nodes, if applicable
        entity_ids[sd] = {0: []}

        if degree > 1:
            num_internal_nodes = len(Ned_at_qpts)
            entity_ids[sd][0] = list(range(cur, cur + num_internal_nodes))

        super(BDMDualSet, self).__init__(nodes, ref_el, entity_ids)


class BrezziDouglasMarini(finite_element.CiarletElement):
    """
    The BDM element

    :arg ref_el: The reference element.
    :arg k: The degree.
    :arg variant: optional variant specifying the types of nodes.

    variant can be chosen from ["point", "integral", "integral(quadrature_degree)"]
    "point" -> dofs are evaluated by point evaluation. Note that this variant has suboptimal
    convergence order in the H(div)-norm
    "integral" -> dofs are evaluated by quadrature rule. The quadrature degree is chosen to integrate
    polynomials of degree 5*k so that most expressions will be interpolated exactly. This is important
    when you want to have (nearly) divergence-preserving interpolation.
    "integral(quadrature_degree)" -> dofs are evaluated by quadrature rule of degree quadrature_degree
    """

    def __init__(self, ref_el, k, variant=None):

        (variant, quad_deg) = check_format_variant(variant, k, "BDM")

        if k < 1:
            raise Exception("BDM_k elements only valid for k >= 1")

        sd = ref_el.get_spatial_dimension()
        poly_set = polynomial_set.ONPolynomialSet(ref_el, k, (sd, ))
        dual = BDMDualSet(ref_el, k, variant, quad_deg)
        formdegree = sd - 1  # (n-1)-form
        super(BrezziDouglasMarini, self).__init__(poly_set, dual, k, formdegree,
                                                  mapping="contravariant piola")
