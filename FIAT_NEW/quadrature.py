import reference_element, expansions, jacobi
import math
import numpy
from factorial import factorial

class QuadratureRule:
    """General class that models integration over a reference element
    as the weighted sum of a function evaluated at a set of points."""
    def __init__( self , ref_el , pts , wts ):
        self.ref_el = ref_el
        self.pts = pts
        self.wts = wts
        return
    def get_points( self ):
        return numpy.array(self.pts)
    def get_weights( self ):
        return numpy.array(self.wts)
    def integrate( self , f ):
        return sum( [ w * f(x) for (x, w) in zip(self.pts, self.wts) ] )

class GaussJacobiQuadratureLineRule( QuadratureRule ):
    """Gauss-Jacobi quadature rule determined by Jacobi weights a and b
    using m roots of m:th order Jacobi polynomial."""
    def __init__( self , ref_el , a , b , m ):
        # this gives roots on the default (-1,1) reference element
        (xs_ref,ws_ref) = compute_gauss_jacobi_rule( a , b , m )

        Ref1 = reference_element.DefaultLine()
        A,b = reference_element.make_affine_mapping( Ref1.get_vertices , \
                                                     ref_el.get_vertices )

        mapping = lambda x: numpy.dot( A , x ) + b

        scale = numpy.linalg.det( A ) * 0.5

        xs = tuple( [ tuple( mapping( x_ref ) ) for x_ref in xs_ref ] )
        ws = tuple( [ scale * w for w in ws_ref ] )

        QuadratureRule.__init__( self , ref_el , xs , ws )

        return


class CollapsedQuadratureTriangleRule( QuadratureRule ):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the square to the triangle."""
    def __init__( self , ref_el , m ):
        ptx,wx = compute_gauss_jacobi_rule(0.,0.,m)
        pty,wy = compute_gauss_jacobi_rule(1.,0.,m)

        # map ptx , pty
        pts_ref = [ expansions.xi_triangle( (x , y) ) \
                    for x in ptx for y in pty ]

        Ref1 = reference_element.DefaultTriangle()
        A,b = reference_element.make_affine_mapping( Ref1.get_vertices() , \
                                                     ref_el.get_vertices() )
        mapping = lambda x: numpy.dot( A , x ) + b

        scale = numpy.linalg.det( A )

        pts = tuple( [ tuple( mapping( x ) ) for x in pts_ref ] )

        wts = [ 0.5 * scale * w1 * w2 for w1 in wx for w2 in wy ]

        QuadratureRule.__init__( self , ref_el , tuple( pts ) , tuple( wts ) )

        return

class CollapsedQuadratureTetrahedronRule( QuadratureRule ):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the cube to the tetrahedron."""
    def __init__( self , ref_el , m ):
        ptx,wx = compute_gauss_jacobi_rule(0.,0.,m)
        pty,wy = compute_gauss_jacobi_rule(1.,0.,m)
        ptz,wz = compute_gauss_jacobi_rule(2.,0.,m)

        # map ptx , pty
        pts_ref = [ expansions.xi_tetrahedron( (x , y, z ) ) \
                    for x in ptx for y in pty for z in ptz ]

        Ref1 = reference_element.DefaultTetrahedron()
        A,b = reference_element.make_affine_mapping( Ref1.get_vertices() , \
                                                     ref_el.get_vertices() )
        mapping = lambda x: numpy.dot( A , x ) + b

        scale = numpy.linalg.det( A )

        pts = tuple( [ tuple( mapping( x ) ) for x in pts_ref ] )

        wts = [ scale * 0.125 * w1 * w2 * w3 \
                for w1 in wx for w2 in wy for w3 in wz ]

        QuadratureRule.__init__( self , ref_el , tuple( pts ) , tuple( wts ) )

        return

def make_quadrature( ref_el , m ):
    """Returns the collapsed quadrature rule using m points per
    direction on the given reference element."""
    if ref_el.get_shape() == reference_element.LINE:
        return GaussJacobiQuadratureLineRule( ref_el , m )
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return CollapsedQuadratureTriangleRule( ref_el , m )
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return CollapsedQuadratureTetrahedronRule( ref_el , m )

# rule to get Gauss-Jacobi points
def compute_gauss_jacobi_points( a , b , m ):
    """Computes the m roots of P_{m}^{a,b} on [-1,1] by Newton's method.
    The initial guesses are the Chebyshev points.  Algorithm
    implemented in Python from the pseudocode given by Karniadakis and
    Sherwin"""
    x = []
    eps = 1.e-8
    max_iter = 100
    for k in range(0,m):
        r = -math.cos(( 2.0*k + 1.0) * math.pi / ( 2.0 * m ) )
        if k > 0:
            r = 0.5 * ( r + x[k-1] )
        j = 0
        delta = 2 * eps
        while j < max_iter:
            s = 0
            for i in range(0,k):
                s = s + 1.0 / ( r - x[i] )
            f = jacobi.eval_jacobi(a,b,m,r)
            fp = jacobi.eval_jacobi_deriv(a,b,m,r)
            delta = f / (fp - f * s)

            r = r - delta

            if math.fabs(delta) < eps:
                break
            else:
                j = j + 1

        x.append(r)
    return x

def compute_gauss_jacobi_rule( a , b , m ):
    xs = compute_gauss_jacobi_points( a , b , m )

    a1 = math.pow(2,a+b+1)
    a2 = gamma(a + m + 1)
    a3 = gamma(b + m + 1)
    a4 = gamma(a + b + m + 1)
    a5 = factorial(m)
    a6 = a1 * a2 * a3 / a4 / a5

    ws = [ a6 / (1.0 - x**2.0) / jacobi.eval_jacobi_deriv(a,b,m,x)**2.0 \
           for x in xs ]

    return xs , ws


# A C implementation for ln_gamma function taken from Numerical
# recipes in C: The art of scientific
# computing, 2nd edition, Press, Teukolsky, Vetterling, Flannery, Cambridge
# University press, page 214
# translated into Python by Robert Kirby
# See originally Abramowitz and Stegun's Handbook of Mathematical Functions.

def ln_gamma( xx ):
    cof = [76.18009172947146,\
           -86.50532032941677, \
           24.01409824083091, \
           -1.231739572450155, \
           0.1208650973866179e-2, \
           -0.5395239384953e-5 ]
    y = xx
    x = xx
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for j in range(0,6):
        y = y + 1
        ser += cof[j] / y
    return -tmp + math.log( 2.5066282746310005*ser/x )

def gamma( xx ):
    return math.exp( ln_gamma( xx ) )

if __name__ == "__main__":
    T = reference_element.DefaultTetrahedron()
    Q = make_quadrature( T , 6 )
    es = expansions.get_expansion_set( T )

    qpts = Q.get_points()
    qwts = Q.get_weights()

    phis = es.tabulate( 3 , qpts )

    foo = numpy.array( [ [ sum( [ qwts[k] * phis[i,k] * phis[j,k] \
                                      for k in range( len( qpts ) ) ] )  \
                           for i in range( phis.shape[0] ) ] \
                             for j in range( phis.shape[0] ) ] )

    print qpts
    print qwts
    #print foo
