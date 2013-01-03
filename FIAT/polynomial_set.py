# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
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

# polynomial sets
# basic interface:
# -- defined over some reference element
# -- need to be able to tabulate (jets)
# -- type of entry: could by scalar, numpy array, or object-value
#    (such as symmetric tensors, as long as they can be converted <-->
#    with 1d arrays)
# Don't need the "Polynomial" class we had before, provided that
# we have an interface for defining sets of functionals (moments against
# an entire set of polynomials)

from . import expansions
import numpy
from .functional import index_iterator
import Scientific.Functions.FirstDerivatives as FirstDerivatives

def mis( m , n ):
    """returns all m-tuples of nonnegative integers that sum up to n."""
    if m == 1:
        return [(n,)]
    elif n == 0:
        return [ tuple( [ 0 ] * m ) ]
    else:
        return [ tuple( [ n - i ] + list( foo ) ) \
                 for i in range( n + 1 ) \
                 for foo in mis( m - 1 , i ) ]

# We order coeffs by C_{i,j,k}
# where i is the index into the polynomial set,
# j may be an empty tuple (scalar polynomials)
#   or else a vector/tensor
# k is the expansion function
# so if I have all bfs at a given point x in an array bf,
# then dot( coeffs , bf ) gives the array of bfs

class PolynomialSet:
    """Implements a set of polynomials as linear combinations of an
    expansion set over a reference element.
    ref_el: the reference element
    degree: an order labeling the space
    embedded degree: the degree of polynomial expansion basis that
         must be used to evaluate this space
    coeffs: A numpy array containing the coefficients of the expansion
         basis for each member of the set.  Coeffs is ordered by
         coeffs[i,j,k] where i is the label of the member, k is
         the label of the expansion function, and j is a (possibly
         empty) tuple giving the index for a vector- or tensor-valued
         function.
    """
    def __init__( self , ref_el , degree , embedded_degree , \
                  expansion_set , coeffs , dmats ):
        self.ref_el = ref_el
        self.num_members =  coeffs.shape[0]
        self.degree = degree
        self.embedded_degree = embedded_degree
        self.expansion_set = expansion_set
        self.coeffs = coeffs
        self.dmats = dmats
        return
    def tabulate_new( self , pts ):
        return numpy.dot( self.coeffs , \
                              self.expansion_set.tabulate( self.embedded_degree, \
                                                               pts ) )
    def tabulate( self , pts , jet_order = 0 ):
        """Returns the values of the polynomial set."""
        result = {}
        base_vals = self.expansion_set.tabulate( self.embedded_degree , pts )
        for i in range( jet_order + 1 ):
            alphas = mis( self.ref_el.get_spatial_dimension() , i )
            for alpha in alphas:
                D = form_matrix_product( self.dmats , alpha )
                result[alpha] = numpy.dot( self.coeffs , \
                                           numpy.dot( numpy.transpose( D ) , \
                                                      base_vals ) )
        return result
    def get_expansion_set( self ):
        return self.expansion_set
    def get_coeffs( self ):
        return self.coeffs
    def get_num_members( self ):
        return self.num_members
    def get_degree( self ):
        return self.degree
    def get_embedded_degree( self ):
        return self.embedded_degree
    def get_dmats( self ):
        return self.dmats
    def get_reference_element( self ):
        return self.ref_el
    def get_shape( self ):
        """Returns the shape of phi(x), where () corresponds to
        scalar (2,) a vector of length 2, etc"""
        return self.coeffs.shape[1:-1]
    def take( self , items ):
        """Extracts subset of polynomials given by items."""
        new_coeffs =  numpy.take( self.get_coeffs() , items , 0 )
        return PolynomialSet( self.ref_el , \
                              self.degree , self.embedded_degree , \
                              self.expansion_set , new_coeffs , \
                              self.dmats )
    def to_sympy( self ):
        import sys
        sys.path.append( ".." )
        import FIAT_S, sympy
        syms = FIAT_S.polynomials.make_syms( self.get_reference_element().get_spatial_dimension() )
        ds_nosub = FIAT_S.polynomials.dubs( self.get_embedded_degree() , syms )
        T1 = reference_element.DefaultReferenceElement()
        T2 = self.get_reference_element()
        A,b = reference_element.make_affine_mapping( T2.get_vertices() , T1.get_vertices() )

        if len( self.coeffs.shape ) == 2:
            return [ sympy.Polynomial( sum( [ self.coeffs[i,j] * ds[j] \
                                                  for j in range( self.coeffs.shape[1] ) ] ) ) \
                         for i in range( self.coeffs.shape[0] ) ]


class ONPolynomialSet( PolynomialSet ):
    """Constructs an orthonormal basis out of expansion set by having
    an identity matrix of coefficients.  Can be used to specify ON
    bases  for vector- and tensor-valued sets as well."""
    def __init__( self , ref_el , degree , shape = tuple() ):

        if shape == tuple():
            num_components = 1
        else:
            flat_shape = numpy.ravel( shape )
            num_components = numpy.prod( flat_shape )
        num_exp_functions = expansions.polynomial_dimension( ref_el , degree )
        num_members = num_components * num_exp_functions
        embedded_degree = degree
        expansion_set = expansions.get_expansion_set( ref_el )
        sd = ref_el.get_spatial_dimension()

        # set up coefficients
        coeffs_shape = tuple( [ num_members ] \
                              + list(shape) \
                              + [ num_exp_functions ] )
        coeffs = numpy.zeros( coeffs_shape , "d" )

        # use functional's index_iterator function
        cur_bf = 0

        if shape == tuple():
            coeffs = numpy.eye( num_members )
        else:
            for idx in index_iterator( shape ):
                for exp_bf in range(0,expansions.polynomial_dimension(ref_el,embedded_degree) ):
                    cur_idx = tuple( [cur_bf] + list( idx ) + [ exp_bf ] )
                    coeffs[cur_idx] = 1.0
                    cur_bf += 1

        # construct dmats
        if degree == 0:
            dmats = [ numpy.array( [ [ 0.0 ] ] , "d" ) \
                      for i in range( sd ) ]
        else:
            pts = ref_el.make_points( sd , 0 , degree + sd + 1 )

            v = numpy.transpose( expansion_set.tabulate( degree , pts ) )
            vinv = numpy.linalg.inv( v )

            #dtildes = expansion_set.tabulate_derivs( degree , pts )
            dpts = numpy.array( [ tuple( [FirstDerivatives.DerivVar( x[i] , i ) \
                                 for i in range(len(x) ) ] ) \
                         for x in pts ] )

            dv = expansion_set.tabulate( degree , dpts )
            dtildes = [ [ [ a[1][i] for a in dvrow ] for dvrow in dv ] for i in range(sd ) ]

            dmats = [ numpy.dot( vinv , numpy.transpose( dtilde ) ) \
                      for dtilde in dtildes ]

        PolynomialSet.__init__( self , ref_el ,
                                degree , embedded_degree , \
                                expansion_set , coeffs , dmats )


def project( f , U , Q ):
    """Computes the expansion coefficients of f in terms of the
    members of a polynomial set U.  Numerical integration is performed
    by quadrature rule Q."""
    pts = Q.get_points()
    wts = Q.get_weights()
    f_at_qps = [ f(x) for x in pts ]
    U_at_qps = U.tabulate( pts )
    coeffs = numpy.array( [ sum( wts * f_at_qps * phi ) \
                            for phi in U_at_qps ] )
    return coeffs

def form_matrix_product( mats , alpha ):
    """forms product over mats[i]**alpha[i]"""
    m = mats[0].shape[0]
    result = numpy.eye( m )
    for i in range( len( alpha ) ):
        for j in range( alpha[i] ):
            result = numpy.dot( mats[i] , result )
    return result

def polynomial_set_union_normalized( A , B ):
    """Given polynomial sets A and B, constructs a new polynomial set
    whose span is the same as that of span(A) union span(B).  It may
    not contain any of the same members of the set, as we construct a
    span via SVD."""
    new_coeffs = numpy.array( list( A.coeffs ) + list( B.coeffs ) )
    func_shape = new_coeffs.shape[1:]
    if len( func_shape ) == 1:
        (u,sig,vt) = numpy.linalg.svd( new_coeffs )
        num_sv = len( [ s for s in sig if abs( s ) > 1.e-10 ] )
        coeffs = vt[:num_sv]
    else:
        new_shape0 = new_coeffs.shape[0]
        new_shape1 = numpy.prod( func_shape )
        newshape = (new_shape0,new_shape1)
        nc = numpy.reshape( new_coeffs , newshape )
        (u,sig,vt) = numpy.linalg.svd( nc , 1 )
        num_sv = len( [ s for s in sig if abs( s ) > 1.e-10 ] )

        coeffs = numpy.reshape( vt[:num_sv] , \
                                tuple( [ num_sv ]
                                       + list( func_shape ) ) )

    return PolynomialSet( A.get_reference_element() , \
                          A.get_degree() ,\
                          A.get_embedded_degree() , \
                          A.get_expansion_set() , \
                          coeffs , A.get_dmats() )

if __name__ == "__main__":
    from . import reference_element

    T = reference_element.UFCTriangle()
    U = ONPolynomialSet( T , 2 )

    print(U.coeffs[0:6,0:6])

    pts = T.make_lattice( 3 )

    jet = U.tabulate( pts , 1 )
    for alpha in sorted( jet ):
        print(alpha)
        print(jet[alpha])


#    print U.get_shape()

