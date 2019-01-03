# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
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

import numpy, collections

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

        tshape = self.nodes[0].target_shape
        num_nodes = len(self.nodes)
        es = poly_set.get_expansion_set()
        num_exp = es.get_num_members(poly_set.get_embedded_degree())

        riesz_shape = tuple([num_nodes] + list(tshape) + [num_exp])

        self.mat = numpy.zeros(riesz_shape, "d")

        for i in range(len(self.nodes)):
            self.mat[i][:] = self.nodes[i].to_riesz(poly_set)

        return self.mat

    def to_riesz_foo(self, poly_set):

        tshape = self.nodes[0].target_shape
        num_nodes = len(self.nodes)
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        num_exp = es.get_num_members(poly_set.get_embedded_degree())
        nbf = poly_set.get_num_members()

        riesz_shape = tuple([num_nodes] + list(tshape) + [num_exp])

        self.mat = numpy.zeros(riesz_shape, "d")

        # let's amalgamate the pt_dict and deriv_dicts of all the
        # functionals so we can tabulate the basis functions twice only
        # (once on pts and once on derivatives)

        # Need: dictionary mapping pts to which functionals they come from
        pts_to_ells = collections.OrderedDict()
        dpts_to_ells = collections.OrderedDict()

        for i, ell in enumerate(self.nodes):
            for pt in ell.pt_dict:
                if pt in pts_to_ells:
                    pts_to_ells[pt].append(i)
                else:
                    pts_to_ells[pt] = [i]

        for i, ell in enumerate(self.nodes):
            for pt in ell.deriv_dict:
                if pt in dpts_to_ells:
                    dpts_to_ells[pt].append(i)
                else:
                    dpts_to_ells[pt] = [i]

        # Now tabulate
        pts = list(pts_to_ells.keys())
        bfs = es.tabulate(ed, pts)


        for j, pt in enumerate(pts):
            which_ells = pts_to_ells[pt]

            for k in which_ells:
                pt_dict = self.nodes[k].pt_dict
                wc_list = pt_dict[pt]

                for i in range(bfs.shape[0]):
                    for (w, c) in wc_list:

                        self.mat[k][c][i] += w*bfs[i, j]


        mdo = max([ell.max_deriv_order for ell in self.nodes])
        if mdo > 0:
            dpts = list(dpts_to_ells.keys())
            es_foo = polynomial_set.ONPolynomialSet(self.ref_el, ed)
            dbfs = es_foo.tabulate(dpts, mdo)

            for j, pt in enumerate(dpts):
                which_ells = dpts_to_ells[pt]

                for k in which_ells:
                    dpt_dict = self.nodes[k].deriv_dict
                    wac_list = dpt_dict[pt]

                    for i in range(nbf):
                        for (w, alpha, c) in wac_list:
                            self.mat[k][c][i] += w*dbfs[alpha][i, j]

        #assert numpy.allclose(self.mat, self.to_riesz_old(poly_set))
        return self.mat
