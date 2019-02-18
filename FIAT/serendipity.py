from sympy import *
from FIAT.finite_element import FiniteElement
from FIAT.reference_element import *



x, y, z = symbols('x y z')

X = (x+1, x-1)
Y = (y+1, y-1)
Z = (z+1, z-1)


class Serendipity(FiniteElement):

    def __init__(self, ref_el, degree):


        super(Serendipity, self).__init__(ref_el, dual, order, formdegree)

    def VLambda0(ref_el):

        dim = ref_el.get_spatial_dimension()

        if dim == 1:
            VL = X
        elif dim == 2:
            VL = tuple([a*b for a in X for b in Y])
        elif dim == 3:
            VL = tuple([a*b*c for a in X for b in Y for c in Z])
        else:
            raise IndexError("reference element must be dimension 1, 2 or 3")

        return VL

    def ELambda0(i, ref_el):

        assert i >= 0, 'invalid value of i'

        dim = ref_el.get_spatial_dimension()

        if dim == 1:
            EL
        elif dim == 2:
            EL = tuple([X[0]*X[1]*b*x**i for b in Y] + [Y[0]*Y[1]*a*y**i for a in X])
        elif dim == 3:
            EL = tuple([X[0]*X[1]*b*c*x**i for b in Y for c in Z] + [Y[0]*Y[1]*a*c*y**i for c in Z for a in X ]
                        + [Z[0]*Z[1]*a*b*z**i for a in X for b in Y])
        else:
            raise IndexError("reference element must be dimension 1, 2 or 3")

        return EL

    def FLambda0(i, ref_el):

        assert i >= 4, 'invalid value for i'

        dim = ref_el.get_spatial_dimension()

        if dim == 2:
            FL = tuple([X[0]*X[1]*Y[0]*Y[1]*(x**(i-4-j))*(y**j) for j in range(i-3)])
        elif dim == 3:
            FL = tuple([X[0]*X[1]*Y[0]*Y[1]*(x**(i-4-j))*(y**j)*c for j in range(i-3) for c in Z]
                        + [X[0]*X[1]*Z[0]*Z[1]*(x**(i-4-j))*(z**j)*b for j in range(i-3) for b in Y]
                        + [Y[0]*Y[1]*Z[0]*Z[1]*(y**(i-4-j))*(z**j)*a for j in range(i-3) for a in X])
        else:
            raise IndexError("reference element must be dimension 2 or 3")

        return FL

    def ILambda0(i, ref_el):

        assert i >= 6, 'invalid value for i'

        dim = ref_el.get_spatial_dimension()

        if dim == 3:
            IL = tuple([X[0]*X[1]*Y[0]*Y[1]*Z[0]*Z[1]*(x**(i-6-j))*(y**(j-k))*(z**k) for j in range(i-5) for k in range(j+1)])
        else:
            raise IndexError("reference element must be dimension 3")

        return IL
        
