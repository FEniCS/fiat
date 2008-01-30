# Written by Robert C. Kirby
# Copyright 2005-2006 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Modified 28 April 2006 by RCK to fix error in FacetMoment
# Modified 26 Sept 2005 by RCK to fix FacetDirectionMoment
# Modified 23 Sept 2005
# Last modified 21 April 2005

import numpy, quadrature, polynomial, shapes, functionaltype
from functionaltype import *

def frob(a,b):
    alen = reduce( lambda a,b:a*b , a.shape )
    return numpy.dot( numpy.reshape( a , (alen,)) , \
                        numpy.reshape( b , (alen,)) )

class Functional( object ):
    def __init__( self , U , a , f_type = "Unspecified" ):
        self.U, self.a = U, numpy.asarray( a )
	self.type = functionaltype.Functionaltype(f_type)
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
    def get_type( self ):
	return self.type.get_attributes()

def PointEvaluation( U , pt ):
    return Functional( U , U.eval_all( pt ) ,
                       Functionaltype("PointEvaluation", [pt]))

# batch mode for making point evaluation functionals
def make_point_evaluations( U , pts ):
    uvals = U.tabulate( pts )
    return [ Functional( U , uvals[:,i] ,
                         Functionaltype("PointEvaluation", [pts[i]])) \
	     for i in range(len(pts)) ]

def ComponentPointEvaluation( U , comp , pt ):
    mat = numpy.zeros( U.coeffs.shape[1:] , "d" )
    bvals = U.base.eval_all( pt )
    mat[comp,:] = bvals
    return Functional( U , mat ,
                       Functionaltype("ComponentPointEvaluation",[pt],[comp]))

def DirectionalComponentPointEvaluation( U , dir , pt ):
    mat = numpy.zeros( U.coeffs.shape[1:] , "d" )
    bvals = U.base.eval_all( pt )
    for i in range(len(dir)):
        mat[i,:] = dir[i] * bvals
    return Functional( U , mat ,
                       Functionaltype("DirectionalComponentPointEvaluation",
                                      [pt], [dir]) )

def make_directional_component_batch( U , dir , pts ):
    mat_shape = tuple( [len(pts)] + list( U.coeffs.shape[1:] ) )
    mat = numpy.zeros(mat_shape,"d")
    bvals = U.base.tabulate( pts )
    for p in range(len(pts)):
        bvals_cur = bvals[:,p]
        for i in range(len(dir)):
            mat[p,i,:] = dir[i] * bvals_cur
    return [ Functional(U, mat[p],
                        Functionaltype("DirectionalComponentPointEvaluation",
                                       [pts[p]], [dir])) \
	     for p in range(len(pts)) ]

# batch mode evaluation for normal components to take advantage of
# bulk tabulation
# but this isn't used anywhere else.
# Can it be deleted?
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
    mat = numpy.zeros( mat_shape , "d" )
    cur = 0
    for dir in dirs:
        for pno in range(len(dirs[dir])):
            bvals_cur = bvals[:,cur]
            for i in range(len(dir)):
                mat[cur,i,:] = dir[i] * bvals_cur
            cur += 1

    # need to fix this to include type information
    # may be issue
    # (Marie has not done anything with this wrt type information.)
    return [ Functional( U , mat[i] ) for i in range(len(pts)) ]

    
# Also, how do I create derivatives?
# I want to evaluate the derivatives at certain points.
# That means evaluating the derivatives 
# Generally, I don't have to do this but 6 or twelve times,
# so I'm not optimizing now.
def PointDerivative( U , i , pt ):
    ftype = Functionaltype("PointDerivative", [pt], None, None, [i])
    return Functional( U , [ u.deriv(i)(pt) for u in U ], ftype)

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
    # Marie: Must do something clever with the type here.
    ftype = Functionaltype("IntegralMoment")
    return Functional( U , p.dof , ftype)


# specifies integration of the i:th partial derivative of a member of
# U against p.
# So, we need to construct a vector L[j] = (dphi_j / dx_i , p ).
# dphi_j / dx_i is a polynomial of the same basis with coefficients
# specified by the j:th column of the i:th dmat of U.base.  So
# we can build the entire vector L[j] = U.base.dmats[i][k,j] p[k],
# using summation notation, or else
# the dot product of dmats[i] transposed with p.
def IntegralMomentOfDerivative( U , i , p ):
    # Marie: Must do something clever with the type here.
    ftype = Functionaltype("IntegralMomentofDerivative")
                           
    return Functional(self, U, 
                      numpy.dot( numpy.transpose( U.base.dmats[i] ), p.dof ),
                      ftype)
    pass

# U is a vector-valued polynomial set and p is a scalar-valued function
# over the same base.
# as above, we need to build an array L[i][j] = (dphi_j / dx_i , p )
# that may be contracted with a member of U.
def IntegralMomentOfDivergence( U , p ):
    mat = numpy.zeros( U.coeffs.shape[1:] , "d" )
    for i in range( U.coeffs.shape[1] ):
        mat[i,:] = numpy.dot( numpy.transpose( U.base.dmats[i] ) , \
                                p.dof )
    ftype = Functionaltype("IntegralMomentofDivergence")
    return Functional( U, mat, ftype)


# what is difference in these two FacetMoment implementations?

#def FacetMoment( U , shape , d , e , p ):
#    # p is defined over the reference element
#    # of dimension d and mapped to facet e of dimension d
#    #over facet e of dimension d
#    # U is a scalar-valued space
#    Qref = quadrature.make_quadrature( d , 3 * U.degree() )
#    alpha = shapes.scale_factor( shape , d , e )
#    if shape == shapes.TRIANGLE:
#        ref_pts = [ x[0] for x in Qref.get_points() ]
#    else:
#        ref_pts = Qref.get_points()
#    pts = map( shapes.pt_maps[ shape ][ d ]( e ) , \
#               ref_pts )
#    wts = alpha * Qref.get_weights()
#    us = U.base.tabulate( pts )
#    ps = numpy.array( [ p(x) for x in Qref.get_points() ] )
#    vec = numpy.array( len( U ) , "d" )
#    for i in len( U ):
#        vec[i] = sum( wts * ps * us[i] )
#
#    return Functional( U , vec , ("FacetMoment",(d,e,p)) )

# U is the space, shape is the reference domain
# we want to integrate members of U against p over
# facet e of dimension d.
# p is defined on the d-dimensional reference domain
def FacetMoment( U , shape , d , e , p ):
    Qref = quadrature.make_quadrature( d , 2 * U.degree() )
    alpha = shapes.scale_factor( shape , d , e )
    pts = Qref.get_points()
    wts = Qref.get_weights() / alpha
    pt_map = shapes.pt_maps[ shape ][ d ]( e )
    mapped_pts = numpy.array( [ pt_map( x ) for x in pts ] )
    ps = numpy.array( [ p( x ) for x in Qref.get_points() ] )
    phis = U.base.tabulate( mapped_pts )
    vec = numpy.dot( phis , wts * ps )

    ftype = Functionaltype("FacetMoment", pts, None, wts)
    return Functional( U , vec , ftype )

def FacetDirectionalMoment( U , shape , dir , d , e , p ):
    mat = numpy.zeros( U.coeffs.shape[1:] , "d" )
    Qref = quadrature.make_quadrature( d , 2 * U.degree() )
    alpha = shapes.scale_factor( shape , d , e )
    pts = Qref.get_points()
    wts = Qref.get_weights() / alpha
    pt_map = shapes.pt_maps[ shape ][ d ]( e )
    mapped_pts = numpy.array( [ pt_map( x ) for x in pts ] )
    ps = numpy.array( [ p( x ) for x in Qref.get_points() ] )
    phis = U.base.tabulate( mapped_pts )

    for i in range(len(dir)):
        mat[i,:] = numpy.dot( phis , dir[i] * wts * ps )

    ftype = Functionaltype("FacetDirectionalMoment", pts, [dir], wts)
    return Functional( U , mat , ftype )

