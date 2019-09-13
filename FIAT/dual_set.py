# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy


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
