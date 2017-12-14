# Copyright (C) 2013 Andrew T. T. McRae, 2015-2016 Jan Blechta, and others
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

from itertools import chain

import numpy

from FIAT.finite_element import FiniteElement
from FIAT.dual_set import DualSet
from FIAT.mixed import concatenate_entity_dofs


__all__ = ['EnrichedElement']


class EnrichedElement(FiniteElement):
    """Class implementing a finite element that combined the degrees of freedom
    of two existing finite elements.

    This is an implementation which does not care about orthogonality of
    primal and dual basis.
    """

    def __init__(self, *elements):
        # Firstly, check it makes sense to enrich.  Elements must have:
        # - same reference element
        # - same mapping
        # - same value shape
        if len(set(e.get_reference_element() for e in elements)) > 1:
            raise ValueError("Elements must be defined on the same reference element")
        if len(set(m for e in elements for m in e.mapping())) > 1:
            raise ValueError("Elements must have same mapping")
        if len(set(e.value_shape() for e in elements)) > 1:
            raise ValueError("Elements must have the same value shape")

        # order is at least max, possibly more, though getting this
        # right isn't important AFAIK
        order = max(e.get_order() for e in elements)
        # form degree is essentially max (not true for Hdiv/Hcurl,
        # but this will raise an error above anyway).
        # E.g. an H^1 function enriched with an L^2 is now just L^2.
        if any(e.get_formdegree() is None for e in elements):
            formdegree = None
        else:
            formdegree = max(e.get_formdegree() for e in elements)

        # set up reference element and mapping, following checks above
        ref_el, = set(e.get_reference_element() for e in elements)
        mapping, = set(m for e in elements for m in e.mapping())

        # set up entity_ids - for each geometric entity, just concatenate
        # the entities of the constituent elements
        entity_ids = concatenate_entity_dofs(ref_el, elements)

        # set up dual basis - just concatenation
        nodes = list(chain.from_iterable(e.dual_basis() for e in elements))
        dual = DualSet(nodes, ref_el, entity_ids)

        super(EnrichedElement, self).__init__(ref_el, dual, order, formdegree, mapping)

        # required degree (for quadrature) is definitely max
        self.polydegree = max(e.degree() for e in elements)

        # Store subelements
        self._elements = elements

    def elements(self):
        "Return reference to original subelements"
        return self._elements

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polydegree

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented")

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""

        num_components = numpy.prod(self.value_shape())
        table_shape = (self.space_dimension(), num_components, len(points))

        table = {}
        irange = slice(0)
        for element in self._elements:

            etable = element.tabulate(order, points, entity)
            irange = slice(irange.stop, irange.stop + element.space_dimension())

            # Insert element table into table
            for dtuple in etable.keys():

                if dtuple not in table:
                    if num_components == 1:
                        table[dtuple] = numpy.zeros((self.space_dimension(), len(points)),
                                                    dtype=etable[dtuple].dtype)
                    else:
                        table[dtuple] = numpy.zeros(table_shape,
                                                    dtype=etable[dtuple].dtype)

                table[dtuple][irange][:] = etable[dtuple]

        return table

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        result, = set(e.value_shape() for e in self._elements)
        return result

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented")
