#Working on 3d trimmed serendipity 2 forms now.
from sympy import symbols, legendre, Array, diff
import numpy as np
from FIAT.finite_element import FiniteElement
from FIAT.dual_set import make_entity_closure_ids
from FIAT.polynomial_set import mis
from FIAT.reference_element import compute_unflattening_map, flatten_reference_cube

x, y, z = symbols('x y z')
variables = (x, y, z)
leg = legendre

def triangular_number(n):
    return int((n+1)*n/2)

def choose_ijk_total(degree):
    top = 1
    for i in range(1, 2 + degree + 1):
        top = i * top
    bottom = 1
    for i in range(1, degree + 1):
        bottom = i * bottom
    return int(top /(2 * bottom))


class TrimmedSerendipity(FiniteElement):
    def __init__(self, ref_el, degree, mapping):
        if degree < 1:
            raise Exception("Trimmed serendipity elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        self.fdim = dim
        if dim != 3:
            if dim !=2:
                raise Exception("Trimmed serendipity elements only valid for dimensions 2 and 3")

        flat_topology = flat_el.get_topology()
        entity_ids = {}
        cur = 0

        for top_dim, entities in flat_topology.items():
            entity_ids[top_dim] = {}
            for entity in entities:
                entity_ids[top_dim][entity] = []

        #3-d case.
        if dim == 3:
            entity_ids[3] = {}
            for l in sorted(flat_topology[1]):
                entity_ids[1][l] = list(range(cur, cur + degree))
                cur = cur + degree
            if (degree > 1):
                face_ids = degree
                for k in range(2, degree):
                    face_ids += 2 * (k - 1)
                for j in sorted(flat_topology[2]):
                    entity_ids[2][j] = list(range(cur, cur + face_ids))
                    cur += face_ids
            interior_ids = 0
            for k in range(4, degree):
                interior_ids = interior_ids + 3 * choose_ijk_total(k - 4)
            if (degree > 3):
                if (degree == 4):
                    interior_tilde_ids = 3
                elif (degree == 5):
                    interior_tilde_ids = 8
                elif (degree == 6):
                    interior_tilde_ids = 6 + 3 * (degree - 3)
                else:
                    interior_tilde_ids = 6 + 3 * (degree - 4) 
            else:
                interior_tilde_ids = 0

            entity_ids[3][0] = list(range(cur, cur + interior_ids + interior_tilde_ids))
            cur = cur + interior_ids + interior_tilde_ids
        else:
            for j in sorted(flat_topology[1]):
                entity_ids[1][j] = list(range(cur, cur + degree))
                cur = cur + degree

            if(degree >= 2):
                entity_ids[2][0] = list(range(cur, cur + 2*triangular_number(degree - 2) + degree))
        
            cur += 2*triangular_number(degree - 2) + degree
        formdegree = 1

        entity_closure_ids = make_entity_closure_ids(flat_el, entity_ids)

        super(TrimmedSerendipity, self).__init__(ref_el=ref_el,
                                                 dual=None,
                                                 order=degree,
                                                 formdegree=formdegree,
                                                 mapping=mapping)

        topology = ref_el.get_topology()
        unflattening_map = compute_unflattening_map(topology)
        unflattened_entity_ids = {}
        unflattened_entity_closure_ids = {}

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

    def degree(self):
        return self._degree

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for trimmed serendipity")

    def get_dual_set(self):
        raise NotImplementedError("get_dual_set is not implemented for trimmed serendipity")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for trimmed serendipity")

    def tabulate(self, order, points, entity=None):

        if entity is None:
            entity = (self.ref_el.get_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        points = list(map(transform, points))

        phivals = {}

        for o in range(order+1):
            alphas = mis(self.fdim, o)
            for alpha in alphas:
                try:
                    polynomials = self.basis[alpha]
                except KeyError:
                    zr = tuple([0] * self.fdim)
                    polynomials = diff(self.basis[zr], *zip(variables, alpha))
                    self.basis[alpha] = polynomials
                T = np.zeros((len(polynomials[:, 0]), self.fdim, len(points)))
                for i in range(len(points)):
                    subs = {v: points[i][k] for k, v in enumerate(variables[:self.fdim])}
                    for ell in range(self.fdim):
                        for j, f in enumerate(polynomials[:, ell]):
                            T[j, ell, i] = f.evalf(subs=subs)
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
        return (self.fdim,)

    def dmats(self):
        raise NotImplementedError

    def get_num_members(self, arg):
        raise NotImplementedError

    def space_dimension(self):
        return int(len(self.basis[tuple([0] * self.fdim)])/self.fdim)


class TrimmedSerendipityCurl(TrimmedSerendipity):
    #3d Curl element needs to be fixed entirely.  This should correspond to 1-forms in 3d.
    def __init__(self, ref_el, degree):
        if degree < 1:
            raise Exception("Trimmed serendipity face elements only valid for k >= 1")

        flat_el = flatten_reference_cube(ref_el)
        dim = flat_el.get_spatial_dimension()
        if dim != 2:
            if dim !=3:
                raise Exception("Trimmed serendipity face elements only valid for dimensions 2 and 3")

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

        if dim == 3:
            if degree < 1:
                raise Exception ("Trimmed Serendipity fce elements only valid for k >= 1")
            EL = e_lambda_1_3d(degree, dx, dy, dz, x_mid, y_mid, z_mid)
            if (degree > 1):
                FL = f_lambda_1_3d(degree, dx, dy, dz, x_mid, y_mid, z_mid)
            else:
                FL = ()
            if (degree > 3):
                IL = I_lambda_1_3d(degree, dx, dy, dz, x_mid, y_mid, z_mid)
            else:
                IL = ()

            Sminus_list = EL + FL + IL
            self.basis = {(0, 0, 0): Array(Sminus_list)}
            super(TrimmedSerendipityCurl, self).__init__(ref_el=ref_el, degree=degree, mapping="contravariant piola")
    
        else:
            ##Put all 2 dimensional stuff here.
            ##This part might be good, need to test it.
            if degree < 1:
                raise Exception("Trimmed serendipity face elements only valid for k >= 1")

            #flat_el = flatten_reference_cube(ref_el)
            #verts = flat_el.get_vertices()
        
            EL = e_lambda_1_2d_part_one(degree, dx, dy, x_mid, y_mid)
            if degree >= 2:
                FL = trimmed_f_lambda_2d(degree, dx, dy, x_mid, y_mid)
            else:
                FL = ()

            Sminus_list = EL + FL
            self.basis = {(0, 0): Array(Sminus_list)}
            super(TrimmedSerendipityCurl, self).__init__(ref_el=ref_el, degree=degree, mapping="contravariant piola")


def e_lambda_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid):
    EL = ()
    #IDs associated to edge 0.
    for j in range(0, deg):
        EL += tuple([(0, 0, leg(j, z_mid) * dx[0] * dy[0])])
    #IDs associated to edge 1.
    for j in range(0, deg):
        EL += tuple([(0, 0, leg(j, z_mid) * dx[0] * dy[1])])
    #IDs associated to edge 2.
    for j in range(0, deg):
        EL += tuple([(0, 0, leg(j, z_mid) * dx[1] * dy[0])])
    #IDs associated to edge 3.
    for j in range(0, deg):
        EL += tuple([(0, 0, leg(j, z_mid) * dx[1] * dy[1])])
    #IDs associated edge 4.
    for j in range(0, deg):
        EL += tuple([(0, leg(j, y_mid) * dx[0] * dz[0], 0)])
    #IDs associated to edge 5.
    for j in range(0, deg):
        EL += tuple([(0, leg(j, y_mid) * dx[0] * dz[1], 0)])
    #IDs associated to edge 6.
    for j in range(0, deg):
        EL += tuple([(0, leg(j, y_mid) * dx[1] * dz[0], 0)])
    #IDs associated to edge 7.
    for j in range(0, deg):
        EL += tuple([(0, leg(j, y_mid) * dx[1] * dz[1], 0)])
    #IDs associated to edge 8.
    for j in range(0, deg):
        EL += tuple([(leg(j, x_mid) * dz[0] * dy[0], 0, 0)])
    #IDs associated to edge 9.
    for j in range(0, deg):
        EL += tuple([(leg(j, x_mid) * dz[1] * dy[0], 0, 0)])
    #IDs associated to edge 10.
    for j in range(0, deg):
        EL += tuple([(leg(j, x_mid) * dz[0] * dy[1], 0, 0)])
    #IDs associated to edge 11.
    for j in range(0, deg):
        EL += tuple([(leg(j, x_mid) * dz[1] * dy[1], 0, 0)])
    return EL


#def f_lambda_1_3d_pieces(deg, dx, dy, dz, x_mid, y_mid, z_mid):
#    FLpiece = ()
#    for l in range(0, 2):
#        for j in range(0, deg - 2):
#            k = deg - 2 - j
#            FLpiece += tuple([(leg(j, x_mid) * leg(k, y_mid) * dz[l] * dy[0] * dy[1], 0, 0)] +
#                         [(leg(j, x_mid) * leg(k, z_mid) * dy[l] * dz[0] * dz[1], 0, 0)] +
#                         [(0, leg(j, y_mid) * leg(k, x_mid) * dz[l] * dx[0] * dx[1], 0)] +
#                         [(0, leg(j, y_mid) * leg(k, z_mid) * dx[l] * dz[0] * dz[1], 0)] +
#                         [(0, 0, leg(j, z_mid) * leg(k, y_mid) * dy[l] * dx[0] * dx[1])] +
#                         [(0, 0, leg(j, z_mid) * leg(k, x_mid) * dx[l] * dy[0] * dy[1])])
#    return FLpiece


#def f_lambda_tilde_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid):
#    FTilde = ()
#    for l in range(0, 2):
#        FTilde += tuple([(leg(deg - 2, y_mid) * dz[l] * dy[0] * dy[1], 0, 0)] +
#                        [(leg(deg - 2, z_mid) * dy[l] * dz[0] * dz[1], 0, 0)] +
#                        [(0, leg(deg - 2, x_mid) * dz[l] * dx[0] * dx[1], 0)] +
#                        [(0, leg(deg - 2, z_mid) * dx[l] * dz[0] * dz[1], 0)] +
#                        [(0, 0, leg(deg - 2, x_mid) * dy[l] * dx[0] * dx[1])] +
#                        [(0, 0, leg(deg - 2, y_mid) * dx[l] * dy[0] * dy[1])])
#    for j in range(1, deg - 1):
#        for l in range(0, 2):
#            FTilde += tuple([(leg(j, x_mid) * leg(deg - j - 2, y_mid) * dz[l] * dy[0] * dy[1], -leg(j - 1, x_mid) * leg(deg - j - 1, y_mid) * dz[l] * dx[0] * dx[1], 0)] +
#                            [(0, leg(j, y_mid) * leg(deg - j - 2, z_mid) * dx[l] * dz[0] * dz[1], -leg(j - 1, y_mid) * leg(deg - j - 1, z_mid) * dx[l] * dy[0] * dy[1])])
#    return FTilde


def f_lambda_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid):
    FL = ()
    #IDs associated to face 0.
    #FTilde first.
    FL += tuple([(0, leg(deg - 2, z_mid) * dx[0] * dz[0] * dz[1], 0)])
    FL += tuple([(0, 0, leg(deg - 2, y_mid) * dx[0] * dy[0] * dy[1])])
    for j in range(1, deg - 1):
        FL += tuple([(0, leg(j, y_mid) * leg(deg - j - 2, z_mid) * dx[0] * dz[0] * dz[1], -leg(j - 1, y_mid) * leg(deg - j - 1, z_mid) * dx[0] * dy[0] * dy[1])])
    #F next
    for i in range(2, deg):
        for j in range(0, i - 1):
            k = i - 2 - j
            FL += tuple([(0, leg(j, y_mid) * leg(k, z_mid) * dx[0] * dz[0] * dz[1], 0)])
            FL += tuple([(0, 0, leg(j, z_mid) * leg(k, y_mid) * dx[0] * dy[0] * dy[1])])
    #IDs associated to face 1.
    FL += tuple([(0, leg(deg - 2, z_mid) * dx[1] * dz[0] * dz[1], 0)])
    FL += tuple([(0, 0, leg(deg - 2, y_mid) * dx[1] * dy[0] * dy[1])])
    for j in range(1, deg - 1):
        FL += tuple([(0, leg(j, y_mid) * leg(deg - j - 2, z_mid) * dx[1] * dz[0] * dz[1], -leg(j - 1, y_mid) * leg(deg - j - 1, z_mid) * dx[1] * dy[0] * dy[1])])
    #F next.
    for i in range(2, deg):
        for j in range(0, i - 1):
            k = i - 2 - j
            FL += tuple([(0, leg(j, y_mid) * leg(k, z_mid) * dx[1] * dz[0] * dz[1], 0)])
            FL += tuple([(0, 0, leg(j, z_mid) * leg(k, y_mid) * dx[1] * dy[0] * dy[1])])
    #IDs associated to face 2.
    #FTilde first.
    FL += tuple([(leg(deg - 2, z_mid) * dy[0] * dz[0] * dz[1], 0, 0)])
    FL += tuple([(0, 0, leg(deg - 2, x_mid) * dy[0] * dx[0] * dx[1])])
    for j in range(1, deg - 1):
        FL += tuple([(leg(j, x_mid) * leg(deg - j - 2, z_mid) * dy[0] * dz[0] * dz[1], 0, -leg(j - 1, x_mid) * leg(deg - j - 1, z_mid) * dy[0] * dx[0] * dx[1])])
    #F next.
    for i in range(2, deg):
        for j in range(0, i - 1):
            k = i - 2 - j
            FL += tuple([(leg(j, x_mid) * leg(k, z_mid) * dy[0] * dz[0] * dz[1], 0, 0)])
            FL += tuple([(0, 0, leg(j, z_mid) * leg(k, x_mid) * dy[0] * dx[0] * dx[1])])
    #IDs associated to face 3.
    FL += tuple([(leg(deg - 2, z_mid) * dy[1] * dz[0] * dz[1], 0, 0)])
    FL += tuple([(0, 0, leg(deg - 2, x_mid) * dy[1] * dx[0] * dx[1])])
    for j in range(1, deg - 1):
        FL += tuple([(leg(j, x_mid) * leg(deg - j - 2, z_mid) * dy[1] * dz[0] * dz[1], 0, -leg(j - 1, x_mid) * leg(deg - j - 1, z_mid) * dy[1] * dx[0] * dx[1])])
    #F next.
    for i in range(2, deg):
        for j in range(0, i - 1):
            k = i - 2 - j
            FL += tuple([(leg(j, x_mid) * leg(k, z_mid) * dy[1] * dz[0] * dz[1], 0, 0)])
            FL += tuple([(0, 0, leg(j, z_mid) * leg(k, x_mid) * dy[1] * dx[0] * dx[1])])
    #IDs associated to face 4.
    FL += tuple([(leg(deg - 2, y_mid) * dz[0] * dy[0] * dy[1], 0, 0)])
    FL += tuple([(0, leg(deg - 2, x_mid) * dz[0] * dx[0] * dx[1], 0)])
    for j in range(1, deg - 1):
        FL += tuple([(leg(j, x_mid) * leg(deg - j - 2, y_mid) * dz[0] * dy[0] * dy[1], -leg(j - 1, x_mid) * leg(deg - j - 1, y_mid) * dz[0] * dx[0] * dx[1], 0)])
    for i in range(2, deg):
        for j in range(0, i - 1):
            k = i - 2 - j
            FL += tuple([(leg(j, x_mid) * leg(k, y_mid) * dz[0] * dy[0] * dy[1], 0, 0)])
            FL += tuple([(0, leg(j, y_mid) * leg(k, x_mid) * dz[0] * dx[0] * dx[1], 0)])
    #IDs associated to face 5.
    FL += tuple([(leg(deg - 2, y_mid) * dz[1] * dy[0] * dy[1], 0, 0)])
    FL += tuple([(0, leg(deg - 2, x_mid) * dz[1] * dx[0] * dx[1], 0)])
    for j in range(1, deg - 1):
        FL += tuple([(leg(j, x_mid) * leg(deg - j - 2, y_mid) * dz[1] * dy[0] * dy[1], -leg(j - 1, x_mid) * leg(deg -j - 1, y_mid) * dz[1] * dx[0] * dx[1], 0)])
    for i in range(2, deg):
        for j in range(0, i - 1):
            k = i - 2 - j
            FL += tuple([(leg(j, x_mid) * leg(k, y_mid) * dz[1] * dy[0] * dy[1], 0, 0)])
            FL += tuple([(0, leg(j, y_mid) * leg(k, x_mid) * dz[1] * dx[0] * dx[1], 0)])
    return FL



#def f_lambda_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid):
#    F = ()
#    for i in range(2, deg):
#        F += f_lambda_1_3d_pieces(deg, dx, dy, dz, x_mid, y_mid, z_mid)
#    F += f_lambda_tilde_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid)
#    return F


def I_lambda_1_3d_pieces(deg, dx, dy, dz, x_mid, y_mid, z_mid):
    I = ()
    for j in range(0, deg - 3):
        for k in range(0, deg - 3 - j):
            l = deg - 4 - j - k
            I += tuple([(leg(j, x_mid) * leg(k, y_mid) * leg(l, z_mid) * dy[0] * dy[1] * dz[0] * dz[1], 0, 0)] +
                       [(0, leg(j, x_mid) * leg(k, y_mid) * leg(l, z_mid) * dx[0] * dx[1] * dz[0] * dz[1], 0)] +
                       [(0, 0, leg(j, x_mid) * leg(k, y_mid) * leg(l, z_mid) * dx[0] * dx[1] * dy[0] * dy[1])])
    return I


def I_lambda_tilde_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid):
    ITilde = ()
    if(deg==4):
        ITilde += tuple([(dy[0] * dy[1] * dz[0] * dz[1], 0, 0)] +
                        [(0, dx[0] * dx[1] * dz[0] * dz[1], 0)] +
                        [(0, 0, dx[0] * dx[1] * dy[0] * dy[1])])       
    if(deg > 4):
        ITilde += tuple([(leg(deg - 4, y_mid) * dy[0] * dy[1] * dz[0] * dz[1], 0, 0)] +
                        [(leg(deg - 4, z_mid) * dy[0] * dy[1] * dz[0] * dz[1], 0, 0)] + 
                        [(0, leg(deg - 4, x_mid) * dx[0] * dx[1] * dz[0] * dz[1], 0)] +
                        [(0, leg(deg - 4, z_mid) * dx[0] * dx[1] * dz[0] * dz[1], 0)] +
                        [(0, 0, leg(deg - 4, x_mid) * dx[0] * dx[1] * dy[0] * dy[1])] +
                        [(0, 0, leg(deg - 4, y_mid) * dx[0] * dx[1] * dy[0] * dy[1])])
    for j in range(1, deg - 3):
        ITilde += tuple([(leg(j, x_mid) * leg(deg - j - 4, y_mid) * dy[0] * dy[1] * dz[0] * dz[1], -leg(j - 1, x_mid) * leg(deg - j - 3, y_mid) * dx[0] * dx[1] * dz[0] * dz[1], 0)] +
                        [(leg(j, x_mid) * leg(deg - j - 4, z_mid) * dy[0] * dy[1] * dz[0] * dz[1], 0, -leg(j - 1, x_mid) * leg(deg - j - 3, z_mid) * dx[0] * dx[1] * dy[0] * dy[1])])
        if deg > 5:
            ITilde += tuple([(0, leg(j, y_mid) * leg(deg - j - 4, z_mid) * dx[0] * dx[1] * dz[0] * dz[1], -leg(j - 1, y_mid) * leg(deg - j - 3, y_mid) * dx[0] * dx[1] * dz[0] * dz[1])])
    if (deg == 6):
        ITilde += tuple([(leg(1, y_mid) * leg(1, z_mid) * dy[0] * dy[1] * dz[0] * dz[1], 0, 0)])
        ITilde += tuple([(0, leg(1, x_mid) * leg(1, z_mid) * dx[0] * dx[1] * dz[0] * dz[1], 0)])
        ITilde += tuple([(0, 0, leg(1, x_mid) * leg(1, y_mid) * dx[0] * dx[1] * dy[0] * dy[1])])
    return ITilde


def I_lambda_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid):
    I = ()
    for i in range(4, deg):
        I += I_lambda_1_3d_pieces(i, dx, dy, dz, x_mid, y_mid, z_mid)
    I += I_lambda_tilde_1_3d(deg, dx, dy, dz, x_mid, y_mid, z_mid)
    return I


#Everything for 2-d should work already.
def e_lambda_1_2d_part_one(deg, dx, dy, x_mid, y_mid):
    EL = tuple(
        [(0, -leg(j, y_mid) * dx[0]) for j in range(deg)] +
        [(0, -leg(j, y_mid) * dx[1]) for j in range(deg)] +
        [(-leg(j, x_mid)*dy[0], 0) for j in range(deg)] +
        [(-leg(j, x_mid)*dy[1], 0) for j in range(deg)])

    return EL


def e_lambda_tilde_1_2d_part_two(deg, dx, dy, x_mid, y_mid):
    ELTilde = tuple([(-leg(deg, x_mid) * dy[0],
                      -leg(deg-1, x_mid) * dx[0] * dx[1] / (deg+1))] +
                    [(-leg(deg, x_mid) * dy[1],
                      leg(deg-1, x_mid) * dx[0] * dx[1] / (deg+1))] +
                    [(-leg(deg-1, y_mid) * dy[0] * dy[1] / (deg+1),
                      -leg(deg, y_mid) * dx[0])] +
                    [(leg(deg-1, y_mid) * dy[0] * dy[1] / (deg+1),
                      -leg(deg, y_mid) * dx[1])])
    return ELTilde


def e_lambda_1_2d(deg, dx, dy, x_mid, y_mid):
    EL = e_lambda_1_2d_part_one(deg, dx, dy, x_mid, y_mid)
    ELTilde = e_lambda_tilde_1_2d_part_two(deg, dx, dy, x_mid, y_mid)

    result = EL + ELTilde
    return result


def determine_f_lambda_portions_2d(deg):
    if (deg < 2):
        DegsOfIteration = []
    else:
        DegsOfIteration = []
        for i in range(2, deg):
            DegsOfIteration += [i]

    return DegsOfIteration


def f_lambda_1_2d_pieces(current_deg, dx, dy, x_mid, y_mid):
    if (current_deg == 2):
        FLpiece = [(leg(0, x_mid) * leg(0, y_mid) * dy[0] * dy[1], 0)]
        FLpiece += [(0, leg(0, x_mid) * leg(0, y_mid) * dx[0] * dx[1])]
    else:
        target_power = current_deg - 2
        FLpiece = tuple([])
        for j in range(0, target_power + 1):
            k = target_power - j
            FLpiece += tuple([(leg(j, x_mid) * leg(k, y_mid) * dy[0] * dy[1], 0)])
            FLpiece += tuple([(0, leg(j, x_mid) * leg(k, y_mid) * dx[0] * dx[1])])
    return FLpiece


def f_lambda_1_2d_trim(deg, dx, dy, x_mid, y_mid):
    DegsOfIteration = determine_f_lambda_portions_2d(deg)
    FL = []
    for i in DegsOfIteration:
        FL += f_lambda_1_2d_pieces(i, dx, dy, x_mid, y_mid)
    return tuple(FL)


def f_lambda_1_2d_tilde(deg, dx, dy, x_mid, y_mid):
    FLTilde = tuple([])
    FLTilde += tuple([(leg(deg - 2, y_mid)*dy[0]*dy[1], 0)])
    FLTilde += tuple([(0, leg(deg - 2, x_mid)*dx[0]*dx[1])])
    for k in range(1, deg - 1):
        FLTilde += tuple([(leg(k, x_mid) * leg(deg - k - 2, y_mid) * dy[0] * dy[1], -leg(k - 1, x_mid) * leg(deg - k - 1, y_mid) * dx[0] * dx[1])])

    return tuple(FLTilde)


def trimmed_f_lambda_2d(deg, dx, dy, x_mid, y_mid):
    FL = f_lambda_1_2d_trim(deg, dx, dy, x_mid, y_mid)
    FLT = f_lambda_1_2d_tilde(deg, dx, dy, x_mid, y_mid)
    result = FL + FLT

    return result
