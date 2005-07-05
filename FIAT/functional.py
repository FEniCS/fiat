# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 21 April 2005

import Numeric, quadrature, polynomial, shapes

def frob(a,b):
    alen = reduce( lambda a,b:a*b , a.shape )
    return Numeric.dot( Numeric.reshape( a , (alen,)) , \
                        Numeric.reshape( b , (alen,)) )

class Functional( object ):
    def __init__( self , U , a ):
        self.U, self.a = U, Numeric.asarray( a )
        return
    def __call__( self , p ):
        # need to add type checking to confirm that
        # p is a member of self.U
        return frob( self.a , p.dof )
    def __add__( self , other ):
	# operations on functionals are cute because now
	# they are implemented by vector arithmetic
	if not self.U is other.U:
	    raise RuntimeError, "type mismatch in adding functionals."
	return Functional( self.U , self.dof + other.dof )

def PointEvaluation( U , pt ):
    return Functional( U , U.eval_all( pt ) )

# batch mode for making point evaluation functionals
def make_point_evaluations( U , pts ):
    uvals = U.tabulate( pts )
    return [ Functional( U , uvals[:,i] ) \
	     for i in range(len(pts)) ]

def ComponentPointEvaluation( U , comp , pt ):
    mat = Numeric.zeros( U.coeffs.shape[1:] , "d" )
    bvals = U.base.eval_all( pt )
    mat[comp,:] = bvals
    return Functional( U , mat )

def DirectionalComponentPointEvaluation( U , dir , pt ):
    mat = Numeric.zeros( U.coeffs.shape[1:] , "d" )
    bvals = U.base.eval_all( pt )
    for i in range(len(dir)):
        mat[i,:] = dir[i] * bvals
    return Functional( U , mat )

def make_directional_component_batch( U , dir , pts ):
    mat_shape = tuple( [len(pts)] + list( U.coeffs.shape[1:] ) )
    mat = Numeric.zeros(mat_shape,"d")
    bvals = U.base.tabulate( pts )
    for p in range(len(pts)):
        bvals_cur = bvals[:,p]
        for i in range(len(dir)):
            mat[p,i,:] = dir[i] * bvals_cur
    return [ Functional(U,m) for m in mat ]
# batch mode evaluation for normal components to take advantage of
# bulk tabulation
def make_directional_component_point_evaluations( U , dirs ):
    """U is a VectorPolynomialSet, dirs is a dictionary mapping
    directions (tuples of component) to the list of points at which we
    want to tabulate that directional component.  Returns a list of
    Functional objects."""
    if not isinstance( U , polynomial.VectorPolynomialSet ):
        raise RuntimeError, "Illegal input."
    
    pts = reduce( lambda a,b: a+b , \
                  [ dirs[c] for c in dirs ] )
    bvals = U.base.tabulate( pts )
    # need a matrix to store the functionals
    mat_shape = tuple( [ len(pts) ] + list( U.coeffs.shape[1:] ) )
    mat = Numeric.zeros( mat_shape , "d" )
    cur = 0
    for dir in dirs:
        for pno in range(len(dirs[dir])):
            bvals_cur = bvals[:,cur]
            for i in range(len(dir)):
                mat[cur,i,:] = dir[i] * bvals_cur
            cur += 1

    return [ Functional( U , mat[i] ) for i in range(len(pts)) ]

    
# Also, how do I create derivatives?
# I want to evaluate the derivatives at certain points.
# That means evaluating the derivatives 
# Generally, I don't have to do this but 6 or twelve times,
# so I'm not optimizing now.
def PointDerivative( U , i , pt ):
    return Functional( U , [ u.deriv(i)(pt) for u in U ] )

# Integral moments on the interior are not hard
# l_u (v) = int( u * v ) means we have to dot the vector of
# coefficients of the function u with the vector of coefficients
# of the input function v, provided that our orthogonal expansions are
# all normalized
#
# This is just Parseval's relation -- L2 inner product of functions
# represented in an orthonormal basis is just the dot product of
# the coefficients in that orthonormal basis

def IntegralMoment( U , p ):
    return Functional( U , p.dof )


# specifies integration of the i:th partial derivative of a member of
# U against p.
# So, we need to construct a vector L[j] = (dphi_j / dx_i , p ).
# dphi_j / dx_i is a polynomial of the same basis with coefficients
# specified by the j:th column of the i:th dmat of U.base.  So
# we can build the entire vector L[j] = U.base.dmats[i][k,j] p[k],
# using summation notation, or else
# the dot product of dmats[i] transposed with p.
def IntegralMomentOfDerivative( U , i , p ):
    return Functional( self , \
                       U , \
                       Numeric.dot( Numeric.transpose( U.base.dmats[i] ) , \
                                    p.dof ) )
    pass

# U is a vector-valued polynomial set and p is a scalar-valued function
# over the same base.
# as above, we need to build an array L[i][j] = (dphi_j / dx_i , p )
# that may be contracted with a member of U.
def IntegralMomentOfDivergence( U , p ):
    mat = Numeric.zeros( U.coeffs.shape[1:] , "d" )
    for i in range( U.coeffs.shape[1] ):
        mat[i,:] = Numeric.dot( Numeric.transpose( U.base.dmats[i] ) , \
                                p.dof )
    return Functional( U , mat )

def FacetMoment( U , shape , d , e , p ):
    # p is defined over the reference element
    # of dimension d and mapped to facet e of dimension d
    #over facet e of dimension d
    # U is a scalar-valued space
    Qref = quadrature.make_quadrature( d , 3 * U.degree() )
    alpha = shapes.scale_factor( shape , d , e )
    if shape == shapes.TRIANGLE:
        ref_pts = [ x[0] for x in Qref.get_points() ]
    else:
        ref_pts = Qref.get_points()
    pts = map( shapes.pt_maps[ shape ][ d ][ e ] , \
               ref_pts )
    wts = alpha * Qref.get_weights()
    us = U.base.tabulate( pts )
    ps = Numeric.array( [ p(x) for x in Qref.get_points() ] )
    vec = Numeric.array( len( U ) , "d" )
    for i in len( U ):
        vec[i] = sum( wts * ps * us[i] )

    return Functional( U , vec )



##def FacetNormalMoment( U , shape , d , e , p ):
##    Qref = quadrature.make_quadrature( d , 3 * U.degree() )
##    alpha = shapes.scale_factor( shape , d , e )
##    if shape == shapes.TRIANGLE:
##        ref_pts = [ x[0] for x in Qref.get_points() ]
##    else:
##        ref_pts = Qref.get_points()
##    pts = map( shapes.pt_maps[ shape ][ d ][ e ] , \
##               ref_pts )
##    wts = alpha * Qref.get_weights()
##    normal = Numeric.array( shapes.normals[shape][e] )
##    us = U.base.tabulate( pts )
##    ps = Numeric.array( [ p(x) for x in Qref.get_points() ] )
##    mat = Numeric.zeros( (len(U.base),U.tensor_shape()[0]) , "d" )
##    for i in range( len( us ) ):
##        for j in range( U.tensor_shape()[0] ):
##            mat[i,j] = sum( wts * us[i] * normal[j] * ps )
##
##    return Functional( U , mat )
##    
