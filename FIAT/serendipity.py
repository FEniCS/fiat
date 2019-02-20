from sympy import *
from FIAT.finite_element import FiniteElement
from FIAT import dual_set, reference_element

x, y, z = symbols('x y z')

X = (x+1, x-1)
Y = (y+1, y-1)
Z = (z+1, z-1)


class Serendipity(FiniteElement):

    def __new__(cls, ref_el, degree):
        dim = ref_el.get_spatial_dimension()
        if dim == 1:
            return Lagrange(ref_el, degree)
        elif dim == 0:
            raise IndexError("reference element cannot be dimension 0")

    def __init__(self, ref_el, degree):

        dim = ref_el.get_spatial_dimension()

        VL = v_lambda_0(dim)
        EL = tuple()
        FL = tuple()
        IL = tuple()
        for i in range(degree + 1):
            EL += e_lambda_0(i, dim)
        for i in range(4, degree + 1):
            FL += f_lambda_0(i, dim)
        s_dict = {0: VL,
                  1: EL,
                  2: FL}
        if dim == 3:
            for i in range(6, degree + 1):
                IL += i_lambda_0(i)
                s_dict[3] = IL

        formdegree = 0

        self.polydegree = degree

        super(Serendipity, self).__init__(ref_el, dual=None, degree, formdegree)

    def degree(self):
        return self.polydegree

    def get_nodal_basis(self):
        raise NotImplementedError("get_nodal_basis not implemented for serendipity")

    def get_coeffs(self):
        raise NotImplementedError("get_coeffs not implemented for serendipity")

    def tabulate(self):
        raise NotImplementedError

    def value_shape(self):
        raise NotImplementedError

    def dmats(self):
        raise NotImplementedError

    def get_num_members(self, arg):
        raise NotImplementedError


def v_lambda_0(dim):

    if dim == 2:
        VL = tuple([a*b for a in X for b in Y])
    else:
        VL = tuple([a*b*c for a in X for b in Y for c in Z])

    return VL

def e_lambda_0(i, dim):

    assert i >= 0, 'invalid value of i'

    if dim == 2:
        EL = tuple([X[0]*X[1]*b*x**i for b in Y]
                   + [Y[0]*Y[1]*a*y**i for a in X])
    else:
        EL = tuple([X[0]*X[1]*b*c*x**i for b in Y for c in Z] +
                   [Y[0]*Y[1]*a*c*y**i for c in Z for a in X] +
                   [Z[0]*Z[1]*a*b*z**i for a in X for b in Y])

    return EL

def f_lambda_0(i, dim):

    assert i >= 4, 'invalid value for i'

    if dim == 2:
        FL = tuple([X[0]*X[1]*Y[0]*Y[1]*(x**(i-4-j))*(y**j)
                    for j in range(i-3)])
    else:
        FL = tuple([X[0]*X[1]*Y[0]*Y[1]*(x**(i-4-j))*(y**j)*c
                    for j in range(i-3) for c in Z] +
                   [X[0]*X[1]*Z[0]*Z[1]*(x**(i-4-j))*(z**j)*b
                    for j in range(i-3) for b in Y] +
                   [Y[0]*Y[1]*Z[0]*Z[1]*(y**(i-4-j))*(z**j)*a
                    for j in range(i-3) for a in X])

    return FL

def i_lambda_0(i):

    assert i >= 6, 'invalid value for i'
    assert dim == 3, 'reference element must be dimension 3'

    IL = tuple([X[0]*X[1]*Y[0]*Y[1]*Z[0]*Z[1]*(x**(i-6-j))*(y**(j-k))*(z**k)
                for j in range(i-5) for k in range(j+1)])

    return IL


class SerendipityDualSet(dual_set.DualSet):

    def __init__(self, ref_el, degree):
        nodes = []
        entity_ids = {}
        topology = ref_el.get_topology()

        for dim in sorted(topology):
            entity_ids[dim] = {}
            for entity in sorted(topology[dim]):
                entity_ids[dim][entity] = []
        super(SerendipityDualSet, self).__init__(nodes, ref_el, entity_ids)


def S(ref_el, degree):
    return Serendipity(ref_el, degree)
