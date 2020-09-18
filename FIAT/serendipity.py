# Copyright (C) 2019 Cyrus Cheng (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2019

from sympy import symbols, legendre, Array, diff, lambdify
import numpy as np
from FIAT.finite_element import FiniteElement
from FIAT.lagrange import Lagrange
from FIAT.dual_set import make_entity_closure_ids
from FIAT.polynomial_set import mis
from FIAT.reference_element import (compute_unflattening_map,
                                    flatten_reference_cube)
from FIAT.reference_element import make_lattice

from FIAT.pointwise_dual import compute_pointwise_dual

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

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        flat_topology = flat_el.get_topology()

        verts = flat_el.get_vertices()

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

        for top_dim, entities in flat_topology.items():
            entity_ids[top_dim] = {}
            for entity in entities:
                entity_ids[top_dim][entity] = []

        for j in sorted(flat_topology[0]):
            entity_ids[0][j] = [cur]
            cur = cur + 1

        EL += e_lambda_0(degree, dim, dx, dy, dz, x_mid, y_mid, z_mid)

        for j in sorted(flat_topology[1]):
            entity_ids[1][j] = list(range(cur, cur + degree - 1))
            cur = cur + degree - 1

        FL += f_lambda_0(degree, dim, dx, dy, dz, x_mid, y_mid, z_mid)

        for j in sorted(flat_topology[2]):
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
        self.basis_callable = {(0,)*dim: lambdify(variables[:dim], Array(s_list),
                                                  modules="numpy", dummify=True)}
        topology = ref_el.get_topology()
        unflattening_map = compute_unflattening_map(topology)
        unflattened_entity_ids = {}
        unflattened_entity_closure_ids = {}

        entity_closure_ids = make_entity_closure_ids(flat_el, entity_ids)

        for dim, entities in sorted(topology.items()):
            unflattened_entity_ids[dim] = {}
            unflattened_entity_closure_ids[dim] = {}
        for dim, entities in sorted(flat_topology.items()):
            for entity in entities:
                unflat_dim, unflat_entity = unflattening_map[(dim, entity)]
                unflattened_entity_ids[unflat_dim][unflat_entity] = entity_ids[dim][entity]
                unflattened_entity_closure_ids[unflat_dim][unflat_entity] = entity_closure_ids[dim][entity]
        self.entity_ids = unflattened_entity_ids
        self.entity_closure_ids = unflattened_entity_closure_ids
        self._degree = degree
        self.flat_el = flat_el

        self.dual = compute_pointwise_dual(self, unisolvent_pts(ref_el, degree))

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
            entity = (self.ref_el.get_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        points = list(map(transform, points))

        phivals = {}
        dim = self.flat_el.get_spatial_dimension()
        if dim <= 1:
            raise NotImplementedError('no tabulate method for serendipity elements of dimension 1 or less.')
        if dim >= 4:
            raise NotImplementedError('tabulate does not support higher dimensions than 3.')
        points = np.asarray(points)
        npoints, pointdim = points.shape
        for o in range(order + 1):
            alphas = mis(dim, o)
            for alpha in alphas:
                try:
                    callable = self.basis_callable[alpha]
                except KeyError:
                    polynomials = diff(self.basis[(0,)*dim], *zip(variables, alpha))
                    callable = lambdify(variables[:dim], polynomials, modules="numpy", dummify=True)
                    self.basis[alpha] = polynomials
                    self.basis_callable[alpha] = callable
                tabulation = callable(*(points[:, i] for i in range(pointdim)))
                T = np.asarray([np.broadcast_to(tab, (npoints, ))
                                for tab in tabulation])
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
        return len(self.basis[(0,)*self.flat_el.get_spatial_dimension()])


def v_lambda_0(dim, dx, dy, dz):

    if dim == 2:
        VL = [a*b for a in dx for b in dy]
    else:
        VL = [a*b*c for a in dx for b in dy for c in dz]

    return VL


def e_lambda_0(i, dim, dx, dy, dz, x_mid, y_mid, z_mid):

    if dim == 2:
        EL = tuple([-leg(j, y_mid) * dy[0] * dy[1] * a for a in dx for j in range(i-1)] +
                   [-leg(j, x_mid) * dx[0] * dx[1] * b for b in dy for j in range(i-1)])
    else:
        EL = tuple([-leg(j, z_mid) * dz[0] * dz[1] * a * b for b in dx for a in dy for j in range(i-1)] +
                   [-leg(j, y_mid) * dy[0] * dy[1] * a * c for a in dx for c in dz for j in range(i-1)] +
                   [-leg(j, x_mid) * dx[0] * dx[1] * b * c for c in dy for b in dz for j in range(i-1)])

    return EL


def f_lambda_0(i, dim, dx, dy, dz, x_mid, y_mid, z_mid):

    if dim == 2:
        FL = tuple([leg(j, x_mid) * leg(k-4-j, y_mid) * dx[0] * dx[1] * dy[0] * dy[1]
                    for k in range(4, i + 1) for j in range(k-3)])
    else:
        FL = tuple([leg(j, y_mid) * leg(k-4-j, z_mid) * dy[0] * dy[1] * dz[0] * dz[1] * a
                    for a in dx for k in range(4, i + 1) for j in range(k-3)] +
                   [leg(j, z_mid) * leg(k-4-j, x_mid) * dx[0] * dx[1] * dz[0] * dz[1] * b
                    for b in dy for k in range(4, i + 1) for j in range(k-3)] +
                   [leg(j, x_mid) * leg(k-4-j, y_mid) * dx[0] * dx[1] * dy[0] * dy[1] * c
                    for c in dz for k in range(4, i + 1) for j in range(k-3)])

    return FL


def i_lambda_0(i, dx, dy, dz, x_mid, y_mid, z_mid):

    IL = tuple([-leg(l-6-j, x_mid) * leg(j-k, y_mid) * leg(k, z_mid) *
                dx[0] * dx[1] * dy[0] * dy[1] * dz[0] * dz[1]
                for l in range(6, i + 1) for j in range(l-5) for k in range(j+1)])

    return IL


def unisolvent_pts(K, deg):
    flat_el = flatten_reference_cube(K)
    dim = flat_el.get_spatial_dimension()
    if dim == 2:
        return unisolvent_pts_quad(flat_el, deg)
    elif dim == 3:
        return unisolvent_pts_hex(flat_el, deg)
    else:
        raise ValueError("Serendipity only defined for quads and hexes")


def unisolvent_pts_quad(K, deg):
    """Gives a set of unisolvent points for the quad serendipity space of order deg.
    The S element is not dual to these nodes, but a dual basis can be constructed from them."""
    L = K.construct_subelement(1)
    vs = np.asarray(K.vertices)
    pts = [pt for pt in K.vertices]
    Lpts = make_lattice(L.vertices, deg, 1)
    for e in K.topology[1]:
        Fmap = K.get_entity_transform(1, e)
        epts = [tuple(Fmap(pt)) for pt in Lpts]
        pts.extend(epts)
    if deg > 3:
        dx0 = (vs[1, :] - vs[0, :]) / (deg-2)
        dx1 = (vs[2, :] - vs[0, :]) / (deg-2)

        internal_nodes = [tuple(vs[0, :] + dx0 * i + dx1 * j)
                          for i in range(1, deg-2)
                          for j in range(1, deg-1-i)]
        pts.extend(internal_nodes)

    return pts


def unisolvent_pts_hex(K, deg):
    """Gives a set of unisolvent points for the hex serendipity space of order deg.
    The S element is not dual to these nodes, but a dual basis can be constructed from them."""
    L = K.construct_subelement(1)
    F = K.construct_subelement(2)
    vs = np.asarray(K.vertices)
    pts = [pt for pt in K.vertices]
    Lpts = make_lattice(L.vertices, deg, 1)
    for e in K.topology[1]:
        Fmap = K.get_entity_transform(1, e)
        epts = [tuple(Fmap(pt)) for pt in Lpts]
        pts.extend(epts)
    if deg > 3:
        fvs = np.asarray(F.vertices)
        # Planar points to map to each face
        dx0 = (fvs[1, :] - fvs[0, :]) / (deg-2)
        dx1 = (fvs[2, :] - fvs[0, :]) / (deg-2)

        Fpts = [tuple(fvs[0, :] + dx0 * i + dx1 * j)
                for i in range(1, deg-2)
                for j in range(1, deg-1-i)]
        for f in K.topology[2]:
            Fmap = K.get_entity_transform(2, f)
            pts.extend([tuple(Fmap(pt)) for pt in Fpts])
    if deg > 5:
        dx0 = np.asarray([1., 0, 0]) / (deg-4)
        dx1 = np.asarray([0, 1., 0]) / (deg-4)
        dx2 = np.asarray([0, 0, 1.]) / (deg-4)

        Ipts = [tuple(vs[0, :] + dx0 * i + dx1 * j + dx2 * k)
                for i in range(1, deg-4)
                for j in range(1, deg-3-i)
                for k in range(1, deg-2-i-j)]
        pts.extend(Ipts)

    return pts
