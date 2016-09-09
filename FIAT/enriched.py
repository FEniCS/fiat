# Copyright (C) 2013 Andrew T. T. McRae, 2015-2016 Jan Blechta
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
from copy import copy

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

    If any of subelements is detected not being nodal, old
    implementation NodelessEnrichedElement is returned instead.
    """
    def __new__(cls, *elements):
        if not all(e.is_nodal() for e in elements):
            # Non-nodal case; use old NodelessEnrichedElement instead
            return NodelessEnrichedElement(*elements)
        else:
            # Nodal case; this class is used
            return object.__new__(cls)

    def __init__(self, *elements):
        # Extract common data
        ref_el = elements[0].get_reference_element()
        expansion_set = elements[0].get_nodal_basis().get_expansion_set()
        degree = min(e.get_nodal_basis().get_degree() for e in elements)
        embedded_degree = max(e.get_nodal_basis().get_embedded_degree()
                              for e in elements)
        order = max(e.get_order() for e in elements)
        mapping = elements[0].mapping()[0]
        formdegree = None if any(e.get_formdegree() is None for e in elements) \
            else max(e.get_formdegree() for e in elements)
        value_shape = elements[0].value_shape()

        # Sanity check
        assert all(e.get_nodal_basis().get_reference_element() ==
                   ref_el for e in elements)
        assert all(type(e.get_nodal_basis().get_expansion_set()) ==
                   type(expansion_set) for e in elements)
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
        offsets = np.cumsum([0] + [e.space_dimension() for e in elements[:-1]])
        entity_ids = _merge_entity_ids((e.entity_dofs() for e in elements),
                                       offsets)

        # Merge dual bases
        nodes = [node for e in elements for node in e.dual_basis()]
        dual_set = DualSet(nodes, ref_el, entity_ids)

        # FiniteElement constructor adjusts poly_set coefficients s.t.
        # dual_set is really dual to poly_set
        FiniteElement.__init__(self, poly_set, dual_set, order,
                               formdegree=formdegree, mapping=mapping)

        # Store subelements
        self._elements = elements

    def elements(self):
        "Return reference to original subelements"
        return self._elements


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


class NodelessEnrichedElement(FiniteElement):
    """Class implementing a finite element that combined the degrees of freedom
    of two existing finite elements.

    This is an old implementation which does not care about orthogonality of
    primal and dual basis. It is kept for compatibility reasons due to
    difficulties implementing tensor product nodes.
    """

    def __init__(self, *elements):
        assert len(elements) == 2, "NodelessEnrichedElement only implemented for two subelements"
        A, B = elements

        # Firstly, check it makes sense to enrich.  Elements must have:
        # - same reference element
        # - same mapping
        # - same value shape
        if not A.get_reference_element() == B.get_reference_element():
            raise ValueError("Elements must be defined on the same reference element")
        if not A.mapping()[0] == B.mapping()[0]:
            raise ValueError("Elements must have same mapping")
        if not A.value_shape() == B.value_shape():
            raise ValueError("Elements must have the same value shape")

        # Set up constituent elements
        self.A = A
        self.B = B

        # required degree (for quadrature) is definitely max
        self.polydegree = max(A.degree(), B.degree())
        # order is at least max, possibly more, though getting this
        # right isn't important AFAIK
        self.order = max(A.get_order(), B.get_order())
        # form degree is essentially max (not true for Hdiv/Hcurl,
        # but this will raise an error above anyway).
        # E.g. an H^1 function enriched with an L^2 is now just L^2.
        if A.get_formdegree() is None or B.get_formdegree() is None:
            self.formdegree = None
        else:
            self.formdegree = max(A.get_formdegree(), B.get_formdegree())

        # set up reference element and mapping, following checks above
        self.ref_el = A.get_reference_element()
        self._mapping = A.mapping()[0]

        # set up entity_ids - for each geometric entity, just concatenate
        # the entities of the constituent elements
        Adofs = A.entity_dofs()
        Bdofs = B.entity_dofs()
        offset = A.space_dimension()  # number of entities belonging to A
        entity_ids = {}

        for ent_dim in Adofs:
            entity_ids[ent_dim] = {}
            for ent_dim_index in Adofs[ent_dim]:
                entlist = copy(Adofs[ent_dim][ent_dim_index])
                entlist += [c + offset for c in Bdofs[ent_dim][ent_dim_index]]
                entity_ids[ent_dim][ent_dim_index] = entlist

        # set up dual basis - just concatenation
        nodes = A.dual_basis() + B.dual_basis()
        self.dual = DualSet(nodes, self.ref_el, entity_ids)

        # Store subelements
        self._elements = elements

    def elements(self):
        "Return reference to original subelements"
        return self._elements

    @staticmethod
    def is_nodal():
        """True if primal and dual bases are orthogonal. If false,
        dual basis is not implemented or is undefined.

        This implementation returns False!
        """
        return False

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

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        # number of dofs just adds
        return self.A.space_dimension() + self.B.space_dimension()

    def tabulate(self, order, points):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""

        # Again, simply concatenate at the basis-function level
        # Number of array dimensions depends on whether the space
        # is scalar- or vector-valued, so treat these separately.

        Asd = self.A.space_dimension()
        Bsd = self.B.space_dimension()
        Atab = self.A.tabulate(order, points)
        Btab = self.B.tabulate(order, points)
        npoints = len(points)
        vs = self.A.value_shape()
        rank = len(vs)  # scalar: 0, vector: 1

        result = {}
        for index in Atab:
            if rank == 0:
                # scalar valued
                # Atab[index] and Btab[index] look like
                # array[basis_fn][point]
                # We build a new array, which will be the concatenation
                # of the two subarrays, in the first index.

                temp = np.zeros((Asd + Bsd, npoints),
                                dtype=Atab[index].dtype)
                temp[:Asd, :] = Atab[index][:, :]
                temp[Asd:, :] = Btab[index][:, :]

                result[index] = temp
            elif rank == 1:
                # vector valued
                # Atab[index] and Btab[index] look like
                # array[basis_fn][x/y/z][point]
                # We build a new array, which will be the concatenation
                # of the two subarrays, in the first index.

                temp = np.zeros((Asd + Bsd, vs[0], npoints),
                                dtype=Atab[index].dtype)
                temp[:Asd, :, :] = Atab[index][:, :, :]
                temp[Asd:, :, :] = Btab[index][:, :, :]

                result[index] = temp
            else:
                raise NotImplementedError("must be scalar- or vector-valued")
        return result

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return self.A.value_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented")
