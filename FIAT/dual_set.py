# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same at Baylor University.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import collections

from FIAT import polynomial_set


class DualSet(object):
    def __init__(self, nodes, ref_el, entity_ids):
        self.nodes = nodes
        self.ref_el = ref_el
        self.entity_ids = entity_ids

        # Compute the nodes on the closure of each sub_entity.
        self.entity_closure_ids = {}
        for dim, entities in ref_el.sub_entities.items():
            self.entity_closure_ids[dim] = {}

            for e, sub_entities in entities.items():
                ids = []

                for d, se in sub_entities:
                    ids += self.entity_ids[d][se]
                ids.sort()
                self.entity_closure_ids[d][e] = ids

    def get_nodes(self):
        return self.nodes

    def get_entity_closure_ids(self):
        return self.entity_closure_ids

    def get_entity_ids(self):
        return self.entity_ids

    def get_reference_element(self):
        return self.ref_el

    def to_riesz(self, poly_set):
        r"""This method gives the action of the entire dual set
        on each member of the expansion set underlying poly_set.
        Then, applying the linear functionals of the dual set to an
        arbitrary polynomial in poly_set is accomplished by (generalized)
        matrix multiplication.

        For scalar-valued spaces, this produces a matrix
        :\math:`R_{i, j}` such that
        :\math:`\ell_i(f) = \sum_{j} a_j \ell_i(\phi_j)`
        for :\math:`f=\sum_{j} a_j \phi_j`.

        More generally, it will have shape concatenating
        the number of functionals in the dual set, the value shape
        of functions it takes, and the number of members of the
        expansion set.
        """

        # This rather technical code queries the low-level information
        # in pt_dict and deriv_dict
        # for each functional to find out where it evaluates its
        # inputs and/or their derivatives.  Then, it tabulates the
        # expansion set one time for all the function values and
        # another for all of the derivatives.  This circumvents
        # needing to call the to_riesz method of each functional and
        # also limits the number of different calls to tabulate.

        tshape = self.nodes[0].target_shape
        num_nodes = len(self.nodes)
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        num_exp = es.get_num_members(poly_set.get_embedded_degree())

        riesz_shape = tuple([num_nodes] + list(tshape) + [num_exp])

        self.mat = numpy.zeros(riesz_shape, "d")

        # Dictionaries mapping pts to which functionals they come from
        pts_to_ells = collections.OrderedDict()
        dpts_to_ells = collections.OrderedDict()

        for i, ell in enumerate(self.nodes):
            for pt in ell.pt_dict:
                if pt in pts_to_ells:
                    pts_to_ells[pt].append(i)
                else:
                    pts_to_ells[pt] = [i]

            for pt in ell.deriv_dict:
                if pt in dpts_to_ells:
                    dpts_to_ells[pt].append(i)
                else:
                    dpts_to_ells[pt] = [i]

        # Now tabulate the function values
        pts = list(pts_to_ells.keys())
        expansion_values = es.tabulate(ed, pts)

        for j, pt in enumerate(pts):
            which_ells = pts_to_ells[pt]

            for k in which_ells:
                pt_dict = self.nodes[k].pt_dict
                wc_list = pt_dict[pt]

                for i in range(num_exp):
                    for (w, c) in wc_list:
                        self.mat[k][c][i] += w*expansion_values[i, j]

        # Tabulate the derivative values that are needed
        max_deriv_order = max([ell.max_deriv_order for ell in self.nodes])
        if max_deriv_order > 0:
            dpts = list(dpts_to_ells.keys())
            # It's easiest/most efficient to get derivatives of the
            # expansion set through the polynomial set interface.
            # This is creating a short-lived set to do just this.
            expansion = polynomial_set.ONPolynomialSet(self.ref_el, ed)
            dexpansion_values = expansion.tabulate(dpts, max_deriv_order)

            for j, pt in enumerate(dpts):
                which_ells = dpts_to_ells[pt]

                for k in which_ells:
                    dpt_dict = self.nodes[k].deriv_dict
                    wac_list = dpt_dict[pt]

                    for i in range(num_exp):
                        for (w, alpha, c) in wac_list:
                            self.mat[k][c][i] += w*dexpansion_values[alpha][i, j]

        return self.mat


def make_entity_closure_ids(ref_el, entity_ids):
    entity_closure_ids = {}
    for dim, entities in ref_el.sub_entities.items():
        entity_closure_ids[dim] = {}

        for e, sub_entities in entities.items():
            ids = []

            for d, se in sub_entities:
                ids += entity_ids[d][se]
            ids.sort()
            entity_closure_ids[d][e] = ids

    return entity_closure_ids
