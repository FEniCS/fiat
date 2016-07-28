# Copyright (C) 2015 Jan Blechta
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

from __future__ import absolute_import, print_function, division

import numpy as np

from FIAT.polynomial_set import PolynomialSet
from FIAT.dual_set import DualSet
from FIAT.finite_element import FiniteElement

__all__ = ['EnrichedElement']

class EnrichedElement(FiniteElement):
    """Enriched element is a direct sum of a sequence of finite elements.
    Dual basis is reorthogonalized to the primal basis for nodality.

    The following is equivalent:
        * the constructor is well-defined,
        * the resulting element is unisolvent and its basis is nodal,
        * the supplied elements are unisolvent with nodal basis and
          their primal bases are mutually linearly independent,
        * the supplied elements are unisolvent with nodal basis and
          their dual bases are mutually linearly independent.
    """

    def __init__(self, elements):
        # Extract common data
        ref_el = elements[0].get_reference_element()
        expansion_set = elements[0].get_nodal_basis().get_expansion_set()
        #FIXME: What is correct degree?
        degree = max(e.get_nodal_basis().get_degree() for e in elements)
        embedded_degree = max(e.get_nodal_basis().get_embedded_degree()
                              for e in elements)
        order = max(e.get_order() for e in elements)
        mapping = elements[0].mapping()[0]
        formdegree = None if any(e.get_formdegree() is None for e in elements) \
                else max(e.get_formdegree() for e in elements)
        value_shape = elements[0].value_shape()

        # Sanity check
        assert all(e.get_nodal_basis().get_reference_element()
                   == ref_el for e in elements)
        assert all(type(e.get_nodal_basis().get_expansion_set())
                   == type(expansion_set) for e in elements)
        assert all(e_mapping == mapping for e in elements
                   for e_mapping in e.mapping())
        assert all(e.value_shape() == value_shape for e in elements)

        # Merge polynomial sets
        coeffs = _merge_coeffs([e.get_coeffs() for e in elements])
        dmats = _merge_dmats([e.dmats() for e in elements])
        poly_set = PolynomialSet(ref_el,
                                 degree,
                                 embedded_degree,
                                 expansion_set,
                                 coeffs,
                                 dmats)


        # Renumber dof numbers
        dims = [e.space_dimension() for e in elements]
        offsets = _compute_offsets(dims)
        entity_ids = _merge_entity_ids((e.entity_dofs() for e in elements),
                                       offsets)

        # Merge dual bases
        nodes = [node for e in elements for node in e.dual_basis()]
        dual_set = DualSet(nodes, ref_el, entity_ids)

        # FiniteElement constructor adjusts poly_set coefficients s.t.
        # dual_set is really dual to poly_set
        FiniteElement.__init__(self, poly_set, dual_set, order,
                formdegree=formdegree, mapping=mapping)


def _merge_coeffs(coeffss):
    shape0 = sum(c.shape[0] for c in coeffss)
    shape1 = max(c.shape[1] for c in coeffss)
    new_coeffs = np.zeros((shape0, shape1), dtype=coeffss[0].dtype)
    counter = 0
    for c in coeffss:
        rows = c.shape[0]
        cols = c.shape[1]
        new_coeffs[counter:counter+rows, :cols] = c
        counter += rows
    assert counter == shape0
    return new_coeffs

def _merge_dmats(dmatss):
    shape, arg = max((dmats[0].shape, args) for args, dmats in enumerate(dmatss))
    assert shape[0] == shape[1]
    new_dmats = []
    for dim in range(len(dmatss[arg])):
        new_dmats.append(dmatss[arg][dim].copy())
        for dmats in dmatss:
            sl = slice(0, dmats[dim].shape[0]), slice(0, dmats[dim].shape[1])
            assert np.allclose(dmats[dim], new_dmats[dim][sl]), \
                    "dmats of elements to be directly summed are not matching!"
    return new_dmats

def _compute_offsets(dimensions):
    offsets = np.zeros(len(dimensions))
    for i, dim in enumerate(dimensions):
        offsets[i+1:] += dim
    return offsets

def _merge_entity_ids(entity_ids, offsets):
    ret = {}
    for i, ids in enumerate(entity_ids):
        for dim in ids:
            if not ret.get(dim):
                ret[dim] = {}
            for entity in ids[dim]:
                if not ret[dim].get(entity):
                    ret[dim][entity] = []
                ret[dim][entity] += (np.array(ids[dim][entity]) + offsets[i]).tolist()
    return ret


if __name__ == '__main__':
    # TODO: Write a proper test, especially for nodality of basis
    from FIAT.lagrange import Lagrange
    from FIAT.bubble import Bubble
    from FIAT.reference_element import UFCTriangle

    T = UFCTriangle()

    # Non-unisolvent case, should fail
    p3 = Lagrange(T, 3)
    p4 = Lagrange(T, 4)
    try:
        e = EnrichedElement((p3, p4))
    except np.linalg.LinAlgError:
        pass
    else:
        assert False

    # Unisolvent case
    p1 = Lagrange(T, 1)
    b3 = Bubble(T, 3)
    e = EnrichedElement((p1, b3))
