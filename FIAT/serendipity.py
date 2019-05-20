# Copyright (C) 2019 Cyrus Cheng (Imperial College London)
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
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2019

from sympy import symbols, legendre, Array, diff
import numpy as np
from FIAT.finite_element import FiniteElement
from FIAT.lagrange import Lagrange
from FIAT.dual_set import make_entity_closure_ids
from FIAT.polynomial_set import mis

x, y, z = symbols('x y z')
variables = (x, y, z)
leg = legendre


def tr(n):
    if n <= 1:
        return 0
    else:
        return int((n-3)*(n-2)/2)


class Serendipity(FiniteElement):

    def __new__(cls, ref_el, degree):
        dim = ref_el.get_spatial_dimension()
        if dim == 1:
            return Lagrange(ref_el, degree)
        elif dim == 0:
            raise IndexError("reference element cannot be dimension 0")
        else:
            self = super().__new__(cls)
            return self

    def __init__(self, ref_el, degree):

        dim = ref_el.get_spatial_dimension()
        topology = ref_el.get_topology()

        x, y, z = symbols('x y z')
        verts = ref_el.get_vertices()

        dx = ((verts[-1][0] - x)/(verts[-1][0] - verts[0][0]), (x - verts[0][0])/(verts[-1][0] - verts[0][0]))
        dy = ((verts[-1][1] - y)/(verts[-1][1] - verts[0][1]), (y - verts[0][1])/(verts[-1][1] - verts[0][1]))
        x_mid = 2*x-(verts[-1][0] + verts[0][0])
        y_mid = 2*y-(verts[-1][1] + verts[0][1])
        try:
            dz = ((verts[-1][2] - z)/(verts[-1][2] - verts[0][2]), (z - verts[0][2])/(verts[-1][2] - verts[0][2]))
            z_mid = 2*z-(verts[-1][2] + verts[0][2])
        except IndexError:
            dz = None
            z_mid = None

        VL = v_lambda_0(dim, dx, dy, dz)
        EL = []
        FL = []
        IL = []
        s_list = []
        entity_ids = {}
        cur = 0

        for top_dim, entities in topology.items():
            entity_ids[top_dim] = {}
            for entity in entities:
                entity_ids[top_dim][entity] = []

        for j in sorted(topology[0]):
            entity_ids[0][j] = [cur]
            cur = cur + 1

        EL += e_lambda_0(degree, dim, dx, dy, dz, x_mid, y_mid, z_mid)

        for j in sorted(topology[1]):
            entity_ids[1][j] = list(range(cur, cur + degree - 1))
            cur = cur + degree - 1

        FL += f_lambda_0(degree, dim, dx, dy, dz, x_mid, y_mid, z_mid)

        for j in sorted(topology[2]):
            entity_ids[2][j] = list(range(cur, cur + tr(degree)))
            cur = cur + tr(degree)

        if dim == 3:
            IL += i_lambda_0(degree, dx, dy, dz, x_mid, y_mid, z_mid)

            entity_ids[3] = {}
            entity_ids[3][0] = list(range(cur, cur + len(IL)))
            cur = cur + len(IL)

        s_list = VL + EL + FL + IL
        assert len(s_list) == cur
        formdegree = 0

        super(Serendipity, self).__init__(ref_el=ref_el, dual=None, order=degree, formdegree=formdegree)

        self.basis = {(0,)*dim: Array(s_list)}
        self.entity_ids = entity_ids
        self.entity_closure_ids = make_entity_closure_ids(ref_el, entity_ids)
        self._degree = degree

    def degree(self):
        return self._degree + 1

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for serendipity")

    def get_dual_set(self):
        raise NotImplementedError("get_dual_set is not implemented for serendipity")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for serendipity")

    def tabulate(self, order, points, entity=None):

        if entity is None:
            entity = (self.ref_el.get_spatial_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        points = list(map(transform, points))

        phivals = {}
        dim = self.ref_el.get_spatial_dimension()
        if dim <= 1:
            raise NotImplementedError('no tabulate method for serendipity elements of dimension 1 or less.')
        if dim >= 4:
            raise NotImplementedError('tabulate does not support higher dimensions than 3.')
        for o in range(order + 1):
            alphas = mis(dim, o)
            for alpha in alphas:
                try:
                    poly = self.basis[alpha]
                except KeyError:
                    poly = diff(self.basis[(0,)*dim], *zip(variables, alpha))
                    self.basis[alpha] = poly
                T = np.zeros((len(poly), len(points)))
                for i in range(len(points)):
                    subs = {v: points[i][k] for k, v in enumerate(variables[:dim])}
                    for j, f in enumerate(poly):
                        T[j, i] = f.evalf(subs=subs)
                phivals[alpha] = T

        return phivals

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
        return len(self.basis[(0,)*self.ref_el.get_spatial_dimension()])


def v_lambda_0(dim, dx, dy, dz):

    if dim == 2:
        VL = [a*b for a in dx for b in dy]
    else:
        VL = [a*b*c for a in dx for b in dy for c in dz]

    return VL


def e_lambda_0(i, dim, dx, dy, dz, x_mid, y_mid, z_mid):

    if dim == 2:
        EL = tuple([-leg(j, y_mid) * dy[0] * dy[1] * a for a in dx for j in range(i-1)]
                   + [-leg(j, x_mid) * dx[0] * dx[1] * b for b in dy for j in range(i-1)])
    else:
        EL = tuple([-leg(j, x_mid) * dx[0] * dx[1] * b * c for b in dy for c in dz for j in range(i-1)]
                   + [-leg(j, y_mid) * dy[0] * dy[1] * a * c for c in dz for a in dx for j in range(i-1)]
                   + [-leg(j, z_mid) * dz[0] * dz[1] * a * b for a in dx for b in dy for j in range(i-1)])

    return EL


def f_lambda_0(i, dim, dx, dy, dz, x_mid, y_mid, z_mid):

    if dim == 2:
        FL = tuple([leg(j, x_mid) * leg(k-4-j, y_mid) * dx[0] * dx[1] * dy[0] * dy[1]
                    for k in range(4, i + 1) for j in range(k-3)])
    else:
        FL = tuple([leg(j, x_mid) * leg(k-4-j, y_mid) * dx[0] * dx[1] * dy[0] * dy[1] * c
                    for k in range(4, i + 1) for j in range(k-3) for c in dz]
                   + [leg(j, z_mid) * leg(k-4-j, x_mid) * dx[0] * dx[1] * dz[0] * dz[1] * b
                      for k in range(4, i + 1) for j in range(k-3) for b in dy]
                   + [leg(j, y_mid) * leg(k-4-j, z_mid) * dy[0] * dy[1] * dz[0] * dz[1] * a
                      for k in range(4, i + 1) for j in range(k-3) for a in dx])

    return FL


def i_lambda_0(i, dx, dy, dz, x_mid, y_mid, z_mid):

    IL = tuple([-leg(l-6-j, x_mid) * leg(j-k, y_mid) * leg(k, z_mid)
                * dx[0] * dx[1] * dy[0] * dy[1] * dz[0] * dz[1]
                for l in range(6, i + 1) for j in range(l-5) for k in range(j+1)])

    return IL
