# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
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
# Modified by Marie E. Rognes (meg@simula.no), 2012

from . import reference_element, expansions, jacobi
import math
import numpy
from .factorial import factorial

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
#    def __init__( self , ref_el , a , b , m ):
    def __init__( self , ref_el , m ):
        # this gives roots on the default (-1,1) reference element
#        (xs_ref,ws_ref) = compute_gauss_jacobi_rule( a , b , m )
        (xs_ref,ws_ref) = compute_gauss_jacobi_rule( 0. , 0. , m )

        Ref1 = reference_element.DefaultLine()
        A,b = reference_element.make_affine_mapping( Ref1.get_vertices() , \
                                                     ref_el.get_vertices() )

        mapping = lambda x: numpy.dot( A , x ) + b

        scale = numpy.linalg.det( A )

        xs = tuple( [ tuple( mapping( x_ref )[0] ) for x_ref in xs_ref ] )
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

class UFCTetrahedronFaceQuadratureRule(QuadratureRule):
    """Highly specialized quadrature rule for the face of a
    tetrahedron, mapped from a reference triangle, used for higher
    order Nedelecs"""
    def __init__(self, face_number, degree):

        # Create quadrature rule on reference triangle
        reference_triangle = reference_element.UFCTriangle()
        reference_rule = make_quadrature(reference_triangle, degree)
        ref_points = reference_rule.get_points()
        ref_weights = reference_rule.get_weights()

        # Get geometry information about the face of interest
        reference_tet = reference_element.UFCTetrahedron()
        face = reference_tet.get_topology()[2][face_number]
        vertices = reference_tet.get_vertices_of_subcomplex(face)

        # Use tet to map points and weights on the appropriate face
        vertices = [numpy.array(list(vertex)) for vertex in vertices]
        x0 = vertices[0]
        J = numpy.matrix([vertices[1] - x0, vertices[2] - x0]).transpose()
        x0 = numpy.matrix(x0).transpose()
        # This is just a very numpyfied way of writing J*p + x0:
        F = lambda p: \
            numpy.array(J*numpy.matrix(p).transpose() + x0).flatten()
        points = numpy.array([F(p) for p in ref_points])

        # Map weights: multiply reference weights by sqrt(|J^T J|)
        detJTJ = numpy.linalg.det(J.transpose()*J)
        weights = numpy.sqrt(detJTJ)*ref_weights

        # Initialize super class with new points and weights
        QuadratureRule.__init__(self, reference_tet, points, weights)
        self._reference_rule = reference_rule
        self._J = J

    def reference_rule(self):
        return self._reference_rule

    def jacobian(self):
        return self._J


def make_quadrature( ref_el , m ):
    """Returns the collapsed quadrature rule using m points per
    direction on the given reference element."""

    msg = "Expecting at least one (not %d) quadrature point per direction" % m
    assert (m > 0), msg
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

#    print qpts
#    print qwts
    #print foo
    cells = [(reference_element.default_simplex(i), reference_element.ufc_simplex(i)) for i in range(1,4)]
    order = 1
    for def_elem, ufc_elem in cells:
        print("\n\ndefault element")
        print(def_elem.get_vertices())
        print("ufc element")
        print(ufc_elem.get_vertices())

        qd = make_quadrature(def_elem, order)
        print("\ndefault points:")
        print(qd.get_points())
        print("default weights:")
        print(qd.get_weights())
        print("sum: ", sum(qd.get_weights()))

        qu = make_quadrature(ufc_elem, order)
        print("\nufc points:")
        print(qu.get_points())
        print("ufc weights:")
        print(qu.get_weights())
        print("sum: ", sum(qu.get_weights()))



