# Written by Robert C. Kirby
# Copyright 2005-2006 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last modified 16 May 2006

"""quadrature.py defines a class for general quadrature rules and then
implements Jacobi-based quadrature rules on lines, triangles, and
tetrahedra.  The primary function to use is the factory
make_quadrature, which takes a shape and the number of points
(per direction in 2d and 3d) and returns the appropriate quadrature
rule object."""


import shapes, jacobi, gamma, factorial, reference
import numpy, math

fact = factorial.factorial

def jacobi_quadrature_rule( a , b , m ):
    """computes (x,w), where x is the list of quadrature points and w is
    the list of quadrature weights.  Algorithm taken from 
    Karniadakis & Sherwin, Appendix B"""
    x = jacobi.compute_gauss_jacobi_points( a , b , m )
    
    w = []
    a1 = math.pow(2,a+b+1);
    a2 = gamma.gamma(a + m + 1);
    a3 = gamma.gamma(b + m + 1);
    a4 = gamma.gamma(a + b + m + 1);
    a5 = fact(m);
    a6 = a1 * a2 * a3 / a4 / a5;
    
    for k in range(0,m):
        fp = jacobi.eval_jacobi_deriv(a,b,m,x[k])
        w.append( a6 / ( 1.0  - x[k]**2.0) / fp ** 2.0 )

    return (x,w)

class QuadratureRule( object ):
    """Base quadrature rule class consisting of points and weights.
    provides a function integrate that integrates a function f.  Can
    also be used as a callable object."""
    def __init__( self, x , w ):
        self.x, self.w = x , numpy.array( w )
        return
    def get_points(self):
        return self.x
    def get_weights(self):
        return self.w
    def __call__( self , f ):
        return self.integrate( f )
    def integrate(self,f):
        fs = numpy.array( [ f( x ) for x in self.x ] )
        return numpy.sum( self.w * fs )

class JacobiQuadrature( QuadratureRule ):
    def __init__( self , a , b , m ):
        #xs,ws = jacobi_quadrature_rule( a , b , m )
        xs,ws = jacobi_quadrature_rule( a , b , m )
        pts = map(reference.xi_line, [(x,) for x in xs])
        scale = reference.get_quadrature_weight_scale(shapes.LINE)
        QuadratureRule.__init__(self, pts, [scale*w for w in ws] ) 
        return

class GaussQuadrature( JacobiQuadrature ):
    """specializes Jacobi quadrature to make the Gauss rules by taking
    the weights both to be zero."""
    def __init__( self , m ):
        JacobiQuadrature.__init__( self , 0. , 0. , m )
        return

class CollapsedQuadratureTriangle( QuadratureRule ):
    """quadrature rule for triangles created by collapsing the points
    on a rectangle."""
    def __init__( self , m ):
        ptx,wx = jacobi_quadrature_rule(0.,0.,m)
        pty,wy = jacobi_quadrature_rule(1.,0.,m)
        scale = reference.get_quadrature_weight_scale(shapes.TRIANGLE)
        ws = numpy.array( [ scale * w1 * w2 for w1 in wx for w2 in wy ] )
        pts = map( reference.xi_triangle, \
                   [ (x,y) for x in ptx for y in pty ] )
        QuadratureRule.__init__( self , pts , ws )
        return

class CollapsedQuadratureTetrahedron( QuadratureRule ):
    """quadrature rule for tetrahedra created by collapsing the points
    on a cube."""
    def __init__( self , m ):
        ptx,wx = jacobi_quadrature_rule(0.,0.,m)
        pty,wy = jacobi_quadrature_rule(1.0,0.0,m)
        ptz,wz = jacobi_quadrature_rule(2.0,0.0,m)
        scale = reference.get_quadrature_weight_scale(shapes.TETRAHEDRON)
        ws = numpy.array( [ scale * w1 * w2 *w3 \
                              for w1 in wx \
                              for w2 in wy \
                              for w3 in wz ] )
        pts = map( reference.xi_tetrahedron, \
                   [ (x,y,z) \
                     for x in ptx \
                     for y in pty \
                     for z in ptz ] )
        QuadratureRule.__init__( self , pts , ws )
        return

def make_quadrature_triangle( m ):
    return CollapsedQuadratureTriangle( m )

def make_quadrature_tetrahedron( m ):
    return CollapsedQuadratureTetrahedron( m )

rules = { shapes.LINE: GaussQuadrature , \
          shapes.TRIANGLE: make_quadrature_triangle, \
          shapes.TETRAHEDRON: make_quadrature_tetrahedron }
  

                             
def make_quadrature( shape , m ):
    """Takes a shape code (see shapes.py) and the number of points
    per direction, and returns the appropriate quadrature rule."""
    if rules.has_key( shape ):
        return rules[shape](m)
    else:
        raise RuntimeError, "No quadrature rule for that shape"

def make_quadrature_by_degree( shape , deg ):
	return make_quadrature(shape, int(math.ceil((deg+1)/2.0)) )

def make_good_quadrature_by_degree( shape , deg ):
	try:
		import newquad
		return newquad.make_quad_by_degree( shape , deg )
	except:
		return make_quadrature_by_degree(shape,deg)
