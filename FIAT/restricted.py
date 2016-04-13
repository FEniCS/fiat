# Copyright (C) 2015 Anders Logg, Marie Rognes, Jan Blechta, Andrew T T McRae
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

# Based on FFC file restrictedelement.py and Jan's own FIAT implementation

from __future__ import absolute_import, print_function, division

import numpy
import six
from six import string_types
from six import iteritems
from .dual_set import DualSet
from .finite_element import CiarletElement


class RestrictedElement(CiarletElement):
    """Restrict given element to specified list of dofs."""

    def __init__(self, element, indices=None, restriction_domain=None):
        '''For sake of argument, indices overrides restriction_domain'''

        if not (indices or restriction_domain):
            raise RuntimeError("Either indices or restriction_domain must be passed in")

        if not indices:
            indices = _get_indices(element, restriction_domain)

        if isinstance(indices, string_types):
            raise RuntimeError("variable 'indices' was a string; did you forget to use a keyword?")

        if len(indices) == 0:
            raise ValueError("No point in creating empty RestrictedElement.")

        self._element = element
        self._indices = indices

        ref_el = element.get_reference_element()

        # Restrict dual set
        dof_counter = 0
        entity_ids = {}
        nodes = []
        nodes_old = element.dual_basis()
        for d, entities in six.iteritems(element.entity_dofs()):
            entity_ids[d] = {}
            for entity, dofs in six.iteritems(entities):
                entity_ids[d][entity] = []
                for dof in dofs:
                    if dof not in indices:
                        continue
                    entity_ids[d][entity].append(dof_counter)
                    dof_counter += 1
                    nodes.append(nodes_old[dof])
        assert dof_counter == len(indices)
        dual = DualSet(nodes, ref_el, entity_ids)
        self.ref_el = ref_el
        self.dual = dual

    def get_reference_element(self):
        return self.ref_el

    def space_dimension(self):
        return len(self._indices)

    def value_shape(self):
        return self._element.value_shape()

    def degree(self):
        return self._element.degree()

    def mapping(self):
        mappings = self._element.mapping()
        return [mappings[i] for i in self._indices]

    def tabulate(self, order, points, entity=None):
        result = self._element.tabulate(order, points, entity)
        extracted = {}
        for (dtuple, values) in sorted_by_key(result):
            extracted[dtuple] = numpy.array([values[i] for i in self._indices])
        return extracted


def sorted_by_key(mapping):
    "Sort dict items by key, allowing different key types."
    # Python3 doesn't allow comparing builtins of different type, therefore the typename trick here
    def _key(x):
        return (type(x[0]).__name__, x[0])
    return sorted(iteritems(mapping), key=_key)


def _get_indices(element, restriction_domain):
    "Restriction domain can be 'interior', 'vertex', 'edge', 'face' or 'facet'"

    if restriction_domain == "interior":
        # Return dofs from interior
        return element.entity_dofs()[max(element.entity_dofs().keys())][0]

    # otherwise return dofs with d <= dim
    if restriction_domain == "vertex":
        dim = 0
    elif restriction_domain == "edge":
        dim = 1
    elif restriction_domain == "face":
        dim = 2
    elif restriction_domain == "facet":
        dim = element.get_reference_element().get_spatial_dimension() - 1
    else:
        raise RuntimeError("Invalid restriction domain")

    is_prodcell = isinstance(max(element.entity_dofs().keys()), tuple)

    entity_dofs = element.entity_dofs()
    indices = []
    for d in range(dim + 1):
        if is_prodcell:
            for a in range(d + 1):
                b = d - a
                try:
                    entities = entity_dofs[(a, b)]
                    for (entity, index) in sorted_by_key(entities):
                        indices += index
                except KeyError:
                    pass
        else:
            entities = entity_dofs[d]
            for (entity, index) in sorted_by_key(entities):
                indices += index
    return indices
