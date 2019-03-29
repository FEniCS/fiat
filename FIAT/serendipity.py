from sympy import *
import numpy as np
from FIAT.finite_element import FiniteElement
from FIAT import dual_set, reference_element
from FIAT.lagrange import Lagrange
from FIAT.dual_set import make_entity_closure_ids
from FIAT.polynomial_set import mis

x, y, z = symbols('x y z')
variables = (x, y, z)


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
        x_mid = 2*(x-(verts[-1][0] + verts[0][0])/2)
        y_mid = 2*(y-(verts[-1][1] + verts[0][1])/2)
        try:
            dz = ((verts[-1][2] - z)/(verts[-1][2] - verts[0][2]), (z - verts[0][2])/(verts[-1][2] - verts[0][2]))
            z_mid = z-(verts[-1][2] + verts[0][2])/2
        except IndexError:
            dz = None
            z_mid = None

        VL = list(v_lambda_0(dim, dx, dy, dz))
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

        for i in range(4, degree + 1):
            FL += f_lambda_0(i, dim, dx, dy, dz, x_mid, y_mid, z_mid)

        for j in sorted(topology[2]):
            entity_ids[2][j] = list(range(cur, cur + int((degree - 3)*(degree - 2)/2)))
            cur = cur + int((degree - 3)*(degree - 2)/2)

        if dim == 3:
            for i in range(6, degree + 1):
                IL += i_lambda_0(i, dx, dy, dz, x_mid, y_mid, z_mid)

            entity_ids[3] = {}
            entity_ids[3][0] = list(range(cur, cur + len(IL)))

        s_list = VL + EL + FL + IL
        formdegree = 0

        super(Serendipity, self).__init__(ref_el=ref_el, dual=None, order=degree, formdegree=formdegree)

        self.basis = {(0,)*dim:Array(s_list)}
        self.entity_ids = entity_ids
        self.entity_closure_ids = make_entity_closure_ids(ref_el, entity_ids)
        self._degree = degree


    def degree(self):
        return self._degree

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
        T = []
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
        VL = tuple([a*b for a in dx for b in dy])
    else:
        VL = tuple([a*b*c for a in dx for b in dy for c in dz])

    return VL

def e_lambda_0(i, dim, dx, dy, dz, x_mid, y_mid, z_mid):

    assert i >= 0, 'invalid value of i'

    if dim == 2:
        EL = tuple([dy[0]*dy[1]*a*y_mid**j for a in dx for j in range(i-1)]
                   + [dx[0]*dx[1]*b*x_mid**j for b in dy for j in range(i-1)])
    else:
        EL = tuple([dx[0]*dx[1]*b*c*x_mid**i for b in dy for c in dz]
                   + [dy[0]*dy[1]*a*c*y_mid**i for c in dz for a in dx]
                   + [dz[0]*dz[1]*a*b*z_mid**i for a in dx for b in dy])

    return EL

def f_lambda_0(i, dim, dx, dy, dz, x_mid, y_mid, z_mid):

    assert i >= 4, 'invalid value for i'

    if dim == 2:
        FL = tuple([dx[0]*dx[1]*dy[0]*dy[1]*(x_mid**(i-4-j))*(y_mid**j)
                    for j in range(i-3)])
    else:
        FL = tuple([dx[0]*dx[1]*dy[0]*dy[1]*(x_mid**(i-4-j))*(y_mid**j)*c
                    for j in range(i-3) for c in dz]
                   + [dx[0]*dx[1]*dz[0]*dz[1]*(x_mid**(i-4-j))*(z_mid**j)*b
                    for j in range(i-3) for b in dy]
                   + [dy[0]*dy[1]*dz[0]*dz[1]*(y_mid**(i-4-j))*(z_mid**j)*a
                    for j in range(i-3) for a in dx])

    return FL

def i_lambda_0(i, dx, dy, dz, x_mid, y_mid, z_mid):

    assert i >= 6, 'invalid value for i'
    assert dim == 3, 'reference element must be dimension 3'

    IL = tuple([dx[0]*dx[1]*dy[0]*dy[1]*dz[0]*dz[1]*(x_mid**(i-6-j))*(y_mid**(j-k))*(z_mid**k)
                for j in range(i-5) for k in range(j+1)])

    return IL
