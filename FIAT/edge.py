# Copyright (C) 2019 Chris Eldred (Inria Grenoble Rhone-Alpes) and Werner Bauer (Inria Rennes)
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
#

import sympy
import numpy as np
from FIAT.finite_element import FiniteElement
from FIAT.dual_set import make_entity_closure_ids
from FIAT.polynomial_set import mis
from FIAT import quadrature

x = sympy.symbols('x')


def _lagrange_poly(i, xi):
    '''Returns the Langrange polynomial P_i(x) of pts xi
    which interpolates the values (0,0,..1,..,0) where 1 is at the ith support point
    :param i: non-zero location
    :param xi: set of N support points
    '''
    index = list(range(len(xi)))
    index.pop(i)
    return sympy.prod([(x-xi[j][0])/(xi[i][0]-xi[j][0]) for j in index])


def _lagrange_basis(spts):
    symbas = []
    for i in range(len(spts)):
        symbas.append(_lagrange_poly(i, spts))
    return symbas


def _create_compatible_l2_basis(cg_symbas):
    ncg_basis = len(cg_symbas)
    symbas = []
    for i in range(1, ncg_basis):
        basis = 0
        for j in range(i):
            basis = basis + sympy.diff(-cg_symbas[j])
        symbas.append(basis)
    return symbas


class _EdgeElement(FiniteElement):

    def __new__(cls, ref_el, degree):
        dim = ref_el.get_spatial_dimension()
        if dim == 1:
            self = super().__new__(cls)
            return self
        else:
            raise IndexError("only intervals supported for _IntervalElement")

    def tabulate(self, order, points, entity=None):

        if entity is None:
            entity = (self.ref_el.get_spatial_dimension(), 0)
        entity_dim, entity_id = entity
        dim = self.ref_el.get_spatial_dimension()

        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        points = list(map(transform, points))

        phivals = {}
        for o in range(order + 1):
            alphas = mis(dim, o)
            for alpha in alphas:
                try:
                    basis = self.basis[alpha]
                except KeyError:
                    basis = sympy.diff(self.basis[(0,)], x)
                    self.basis[alpha] = basis
                T = np.zeros((len(basis), len(points)))
                for i in range(len(points)):
                    subs = {x: points[i][0]}
                    for j, f in enumerate(basis):
                        T[j, i] = f.evalf(subs=subs)
                phivals[alpha] = T

        return phivals

    def degree(self):
        return self._degree

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for _IntervalElement")

    def get_dual_set(self):
        raise NotImplementedError("get_dual_set is not implemented for _IntervalElement")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for _IntervalElement")

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self.entity_ids

    def entity_closure_dofs(self):
        """Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element."""
        return self.entity_closure_ids

    def value_shape(self):
        return ()

    def dmats(self):
        raise NotImplementedError

    def get_num_members(self, arg):
        raise NotImplementedError

    def space_dimension(self):
        return len(self.basis[(0,)])

    def __init__(self, ref_el, degree):

        dim = ref_el.get_spatial_dimension()
        topology = ref_el.get_topology()

        if not dim == 1:
            raise IndexError("only intervals supported for DMSE")

        formdegree = 1

        # This is general code to build empty entity_ids
        entity_ids = {}
        for dim in range(dim+1):
            entity_ids[dim] = {}
            for entity in sorted(topology[dim]):
                entity_ids[dim][entity] = []

        # The only dofs for DMSE are with the interval!
        entity_ids[dim][0] = list(range(degree + 1))

        # Build the basis
        # This is a dictionary basis[o] that gives the "basis" functions for derivative order o, where o=0 is just the basis itself
        # This is filled as needed for o > 0 by tabulate

        self.entity_ids = entity_ids
        self.entity_closure_ids = make_entity_closure_ids(ref_el, entity_ids)
        self._degree = degree

        super(_EdgeElement, self).__init__(ref_el=ref_el, dual=None, order=degree, formdegree=formdegree)


class EdgeGaussLobattoLegendre(_EdgeElement):
    def __init__(self, ref_el, degree):
        super(EdgeGaussLobattoLegendre, self).__init__(ref_el, degree)
        cont_pts = quadrature.GaussLobattoLegendreQuadratureLineRule(ref_el, degree+2).pts
        cont_basis = _lagrange_basis(cont_pts)
        basis = _create_compatible_l2_basis(cont_basis)
        self.basis = {(0,): sympy.Array(basis)}


class EdgeExtendedGaussLegendre(_EdgeElement):
    def __init__(self, ref_el, degree):
        super(EdgeExtendedGaussLegendre, self).__init__(ref_el, degree)
        cont_pts = quadrature.ExtendedGaussLegendreQuadratureLineRule(ref_el, degree+2).pts
        cont_basis = _lagrange_basis(cont_pts)
        basis = _create_compatible_l2_basis(cont_basis)
        self.basis = {(0,): sympy.Array(basis)}
