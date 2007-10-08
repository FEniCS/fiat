# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 4 Feb 2005 by RCK

import jacobi, shapes, math, numpy
from reference import *

"""Principal expansion functions as defined by Karniadakis & Sherwin.
The main point of entry to this module is the function
make_expansion( shape , n ) which takes the simplex shape and the
polynomal degree n and returns the list of expansion functions up to
that degree."""

# We define the principal expansion functions (p.79 K&S). These take
# values z in [-1, 1] and are clever functions of the Jacobi
# polynomials. The factors for triangles and tetrahedra are introduced
# to "keep the expansions are polynomials in terms of the Cartesian
# coordinates (\eta_1, \eta_2, \eta_3)" and are thus not dependent on
# the choice of reference cell.
def psitilde_a( i , z ):
    return jacobi.eval_jacobi( 0 , 0 , i , z )

def psitilde_b( i , j , z ):
    if i == 0:
        return jacobi.eval_jacobi( 1 , 0 , j , z )
    else:
        return (0.5 * ( 1 - z )) ** i * jacobi.eval_jacobi( 2*i+1 , 0 , j , z )

def psitilde_c( i , j , k , z ):
    if i + j == 0:
        return jacobi.eval_jacobi( 2 , 0 , k , z )
    else:
        return ( 0.5 * (1-z))**(i+j) \
               * jacobi.eval_jacobi( 2*(i+j+1), 0 , k , z )

# The expansions $\phi$ i.e the polynomials basis functions are
# defined in terms of the principal basis functions (\tilde \psi^a and
# in 2D/3D \tilde \psi^b and in 3D \tilde \psi_c).

# Example: For the triangle, we have
# \phi_{p,q}(\xi_1, \xi_2) = \tilde \psi^a_p(\eta_1) \tilde \psi^b_q(\eta_2)

def phi_line( index , xi ):
    (p,) = index
    (eta,) = xi
    return psitilde_a( p , eta )

def phi_triangle( index , xi ):
    p,q = index
    eta1,eta2 = eta_triangle( xi )

    alpha = psitilde_a( p , eta1 )
    beta = psitilde_b( p , q , eta2 )

    return alpha * beta

def phi_tetrahedron( index , xi ):
    p,q,r = index
    eta1,eta2,eta3 = eta_tetrahedron( xi )

    alpha = psitilde_a( p , eta1 )
    beta = psitilde_b( p , q , eta2 )
    gamma = psitilde_c( p , q , r , eta3 )

    return alpha * beta * gamma

# Some class to store all this information consistently
class ExpansionFunction( object ):
    """The class ExpansionFunction represent a basis function.

    Attributes:

         indices     -   The basis function index
         phi         -   The basis function phi = phi(indices, xi) 
         alpha       -   A factor

    An ExpansionFunction takes a point xi and returns alpha*phi_indices(xi)
    """
 
    def __init__( self , indices , phi , alpha ):
        self.indices, self.phi,self.alpha = indices , phi , alpha
        return
    def __call__( self , x ):
        if len( x ) != len( self.indices ):
            raise RuntimeError, "Illegal number of coordinates"
        return self.alpha * self.phi( self.indices , x )

class PhiLine( ExpansionFunction ):
    def __init__( self , i ):
        ExpansionFunction.__init__( self , \
                                    (i , ) , \
                                    phi_line , \
                                    math.sqrt(1.0*i+0.5) )

class PhiTriangle( ExpansionFunction ):
    def __init__( self , i , j ):
        ExpansionFunction.__init__( self , \
                                    ( i,j ) , \
                                    phi_triangle, \
                                    math.sqrt( (i+0.5)*(i+j+1.0) ) )
        return

class PhiTetrahedron( ExpansionFunction ):
    def __init__( self , i , j , k ):
        ExpansionFunction.__init__( self , (i,j,k) , \
                                    phi_tetrahedron , \
                                    math.sqrt( (i+0.5)*(i+j+1.0)*(i+j+k+1.5)))
        return

# Generate basis functions of all polynomial orders up to n:
def make_phis_line( n ):
    return [ PhiLine( i ) \
             for i in range(0,n+1) ]

def make_phis_triangle( n ):
    return [ PhiTriangle( k - i , i ) \
             for k in range( 0 , n + 1 ) \
             for i in range( 0 , k + 1 ) ]

def make_phis_tetrahedron( n ):
    return [ PhiTetrahedron( k - i - j , j , i ) \
             for k in range( 0 , n + 1 ) \
             for i in range( 0 , k + 1 ) \
             for j in range( 0 , k - i + 1 ) ]

make_phis = { shapes.LINE : make_phis_line , \
              shapes.TRIANGLE : make_phis_triangle , \
              shapes.TETRAHEDRON : make_phis_tetrahedron }

def make_expansion( shape , n ):
    """Returns the orthogonal expansion basis on a given shape
    for polynomials of degree n."""
    global make_phis
    try:
        return make_phis[shape]( n )
    except:
        raise shapes.ShapeError, "expansions.make_expansion: Illegal shape"



# Now this is the tabulation part. Quite independent of the rest.
# This does the same as the rest I guess, just written out in gory
# detail and is vectorized so that we can evaluate a lot of points at
# once. (Where as the others are designed to evaluate one point at a
# time.) This could probably be cleaned up! One of these should be
# very unnecessary!

# The phis on the line is very simple. No change of variables involved.
def tabulate_phis_line( n , xs ):
    """Tabulates all the basis functions over the line up to
    degree in at points xs."""
    # extract raw points from singletons
    ys = numpy.array( [ x for (x,) in xs ] )
    phis = jacobi.eval_jacobi_batch(0,0,n,ys)
    for i in range(0,len(phis)):
	phis[i,:] *= math.sqrt(1.0*i+0.5)
    return phis

def tabulate_phis_derivs_line( n , xs ):
    ys = numpy.array( [ x for (x,) in xs ] )
    phi_derivs = jacobi.eval_jacobi_deriv_batch(0,0,n,ys)
    for i in range(0,len(xs)):
	phi_derivs[i,:] *= math.sqrt(1.0*i+0.5)
    return (phi_derivs,)

# Gets a bit more messy on the triangle. In particular, we need some
# scalings also known as the factors that were used in front of the
# definitions for psitilde_xxx above. (Add them explicitely here.)

# These scalings are intended to do something. Unknown at present.
def make_scalings( n , etas ):
    # Initialize an (n+1)xlen(etas) zero matrix filled with doubles ("d").
    scalings = numpy.zeros( (n+1,len(etas)) ,"d")
    # Set the first row entries equal to 1
    scalings[0,:] = 1.0
    if n > 0:
        # Let S(n, i) = S(1, i)^n
        # where S(1, i) = 0.5(1.0-etas(i)) for each i
	scalings[1,:] = 0.5 * (1.0 - etas)
	for k in range(2,n+1):
	    scalings[k,:] = scalings[k-1,:] * scalings[1,:]
    return scalings

def tabulate_phis_triangle( n , xs ):
    """Tabulates all the basis functions over the triangle
    up to degree n"""

    # unpack coordinates
    etas = map( eta_triangle , xs )
    eta1s = numpy.array( [ eta1 for (eta1,eta2) in etas ] )
    eta2s = numpy.array( [ eta2 for (eta1,eta2) in etas ] )

    # get Legendre functions for eta1 direction
    psitilde_as = jacobi.eval_jacobi_batch(0,0,n,eta1s)

    # scalings ( (1-eta2) / 2 ) ** i
    scalings = make_scalings( n , eta2s )

    # for i == 0, I can have j == 0...degree
    # for i == 1, I can have j == 0...degree-1
    
    # phi_{i,j} requires p_j^{2i+1,0}
    psitilde_bs = [ jacobi.eval_jacobi_batch(2*i+1,0,n-i,eta2s) \
		    for i in range(0,n+1) ]
	    
    results = numpy.zeros( (shapes.poly_dims[shapes.TRIANGLE](n),len(xs)) , \
			     "d" )
    # Multiply the separate factors together to form phi. Note that
    # the scalings simply is connected with the definition of psi_b
    cur = 0
    for k in range(0,n+1):
	for i in range(0,k+1):
	    ii = k-i
	    jj = i
	    results[cur,:] = psitilde_as[ii,:] * scalings[ii,:] \
		* psitilde_bs[ii][jj,:]
	    results[cur,:] *= math.sqrt( (ii+0.5)*(ii+jj+1.0) )
	    cur += 1

    return results

def tabulate_phis_derivs_triangle(n,xs):
    """I'm not properly handling the singularity at the top,
    so we can't differentiation for eta2 == 0."""
    # unpack coordinates
    etas = map( eta_triangle , xs )
    eta1s = numpy.array( [ eta1 for (eta1,eta2) in etas ] )
    eta2s = numpy.array( [ eta2 for (eta1,eta2) in etas ] )

    # Generate the psis as for the non-derivative case.
    psitilde_as = jacobi.eval_jacobi_batch(0,0,n,eta1s)
    psitilde_derivs_as = jacobi.eval_jacobi_deriv_batch(0,0,n,eta1s)
    psitilde_bs = [ jacobi.eval_jacobi_batch(2*i+1,0,n-i,eta2s) \
		    for i in range(0,n+1) ]
    psitilde_derivs_bs = [ jacobi.eval_jacobi_deriv_batch(2*i+1,0,n-i,eta2s) \
			   for i in range(0,n+1) ]

    # These are the missing factors for psitilde_bs:
    scalings = make_scalings(n, eta2s);
	    
    # Arrays to store the results in:
    xderivs = numpy.zeros( (shapes.poly_dims[shapes.TRIANGLE](n),len(xs)) , \
			     "d" )
    yderivs = numpy.zeros( xderivs.shape , "d" )
    tmp = numpy.zeros( (len(xs),) , "d" )

    # The following block differentiates the expansion function phi
    # (using the chain rule).
    scale = get_chain_rule_scaling()
    cur = 0
    for k in range(0,n+1):
	for i in range(0,k+1):
	    ii = k-i
	    jj = i

            # Differentiate using the chain rule w.r.t x (aka xi1):
            xderivs[cur,:] = psitilde_derivs_as[ii,:]*psitilde_bs[ii][jj,:]
	    if ii > 0:
		xderivs[cur,:] *= scalings[ii-1,:]

            # Differentiate using the chain rule w.r.t y (aka xi2).
            # Term 1: deta1/dxi2 * d/deta1(psitilde_a(eta1)*psitilde_b(eta2))
	    yderivs[cur,:] = psitilde_derivs_as[ii,:]*psitilde_bs[ii][jj,:] \
                             *0.5*(1.0+eta1s[:])
	    if ii > 0:
		yderivs[cur,:] *= scalings[ii-1,:]

            # Term 2: deta2/dxi2 * d/deta2(psitilde_a(eta1)*psitilde_b(eta2))
	    tmp[:] = psitilde_derivs_bs[ii][jj,:]*scalings[ii,:]
	    if ii > 0:
		tmp[:] -= 0.5 * ii * psitilde_bs[ii][jj,:]*scalings[ii-1]

	    yderivs[cur,:] += psitilde_as[ii,:]*tmp
	    
            # Normalize with alpha:
	    alpha = math.sqrt( (ii+0.5)*(ii+jj+1.0) )
	    xderivs[cur,:] *= scale*alpha
	    yderivs[cur,:] *= scale*alpha
	    cur += 1

    return (xderivs,yderivs)

def tabulate_phis_tetrahedron( n , xs ):
    """Tabulates all the basis functions over the tetrahedron
    up to degree n"""
    # unpack coordinates
    etas = map( eta_tetrahedron , xs )
    eta1s = numpy.array( [ eta1 for (eta1,eta2,eta3) in etas ] )
    eta2s = numpy.array( [ eta2 for (eta1,eta2,eta3) in etas ] )
    eta3s = numpy.array( [ eta3 for (eta1,eta2,eta3) in etas ] )

    # get Legendre functions for eta1 direction
    psitilde_as = jacobi.eval_jacobi_batch(0,0,n,eta1s)
    eta2_scalings = make_scalings(n, eta2s)

    psitilde_bs = [ jacobi.eval_jacobi_batch(2*i+1,0,n-i,eta2s) \
		    for i in range(0,n+1) ]
    eta3_scalings = make_scalings(n, eta3s)

    # I need psitilde_c[i][j][k]
    # for 0<=i+j+k<=n
    psitilde_cs = [ [ jacobi.eval_jacobi_batch(2*(i+j+1),0,\
					       n-i-j,eta3s) \
		      for j in range(0,n+1-i) ] for i in range(0,n+1) ]

    results = numpy.zeros( (shapes.poly_dims[shapes.TETRAHEDRON](n), \
			      len(xs)) , \
			     "d" )
    cur = 0
    for k in range(0,n+1):  # loop over degree
	for i in range(0,k+1):
	    for j in range(0,k-i+1):
		ii = k-i-j
		jj = j
		kk = i
		results[cur,:] = psitilde_as[ii,:] * \
		    eta2_scalings[ii,:] \
		    * psitilde_bs[ii][jj,:] \
		    * eta3_scalings[ii+jj,:] \
		    * psitilde_cs[ii][jj][kk,:]
		results[cur,:] *= math.sqrt( (ii+0.5) \
					     * (ii+jj+1.0) \
					     * (ii+jj+kk+1.5) )
		cur += 1

    return results

def tabulate_phis_derivs_tetrahedron(n,xs):
    """Tabulates all the derivatives of basis functions over the tetrahedron
    up to degree n at points xs"""
    # unpack coordinates
    etas = map( eta_tetrahedron , xs )
    eta1s = numpy.array( [ eta1 for (eta1,eta2,eta3) in etas ] )
    eta2s = numpy.array( [ eta2 for (eta1,eta2,eta3) in etas ] )
    eta3s = numpy.array( [ eta3 for (eta1,eta2,eta3) in etas ] )

    psitilde_as = jacobi.eval_jacobi_batch(0,0,n,eta1s)
    psitilde_as_derivs = jacobi.eval_jacobi_deriv_batch(0,0,n,eta1s)
    psitilde_bs = [ jacobi.eval_jacobi_batch(2*i+1,0,n-i,eta2s) \
		    for i in range(0,n+1) ]
    psitilde_bs_derivs = [ jacobi.eval_jacobi_deriv_batch(2*i+1,0,n-i,eta2s) \
                           for i in range(0,n+1) ]
    psitilde_cs = [ [ jacobi.eval_jacobi_batch(2*(i+j+1),0,\
					       n-i-j,eta3s) \
		      for j in range(0,n+1-i) ] for i in range(0,n+1) ]
    psitilde_cs_derivs = [ [ jacobi.eval_jacobi_deriv_batch(2*(i+j+1),0,\
                                                            n-i-j,eta3s) \
                             for j in range(0,n+1-i) ] for i in range(0,n+1) ]
    
    eta2_scalings = make_scalings(n, eta2s)
    eta3_scalings = make_scalings(n, eta3s)

    tmp = numpy.zeros( (len(xs),) , "d" )
    xderivs = numpy.zeros( (shapes.poly_dims[shapes.TETRAHEDRON](n), \
			      len(xs)) , \
			     "d" )
    yderivs = numpy.zeros( xderivs.shape , "d" )
    zderivs = numpy.zeros( xderivs.shape , "d" )
    scale = get_chain_rule_scaling()

    cur = 0
    for k in range(0,n+1):  # loop over degree
	for i in range(0,k+1):
	    for j in range(0,k-i+1):
		ii = k-i-j
		jj = j
		kk = i

		xderivs[cur,:] = psitilde_as_derivs[ii,:]
		xderivs[cur,:] *= psitilde_bs[ii][jj,:]
		xderivs[cur,:] *= psitilde_cs[ii][jj][kk,:]
		if ii>0:
		    xderivs[cur,:] *= eta2_scalings[ii-1]
		if ii+jj>0:
		    xderivs[cur,:] *= eta3_scalings[ii+jj-1]

		#d/deta1 with scalings
		yderivs[cur,:] = psitilde_as_derivs[ii,:]
		yderivs[cur,:] *= psitilde_bs[ii][jj,:]
		yderivs[cur,:] *= psitilde_cs[ii][jj][kk,:]
		yderivs[cur,:] *= 0.5 * (1.0+eta1s[:])
		if ii>0:
		    yderivs[cur,:] *= eta2_scalings[ii-1]
		if ii+jj>0:
		    yderivs[cur,:] *= eta3_scalings[ii+jj-1]
		
		#tmp will hold d/deta2 term
		tmp[:] = psitilde_bs_derivs[ii][jj,:]
		tmp[:] *= eta2_scalings[ii,:]
		if ii>0:
		    tmp[:] -= 0.5*ii*eta2_scalings[ii-1]*psitilde_bs[ii][jj,:]
		tmp[:] *= psitilde_as[ii,:]
		tmp[:] *= psitilde_cs[ii][jj][kk,:]
		if ii+jj>0:
		    tmp[:] *= eta3_scalings[ii+jj-1]

                # Adding the two terms, gives the final yderivs
		yderivs[cur,:] += tmp[:]

		# zderivative
		#d/deta1 with scalings
		zderivs[cur,:] = psitilde_as_derivs[ii,:]
		zderivs[cur,:] *= psitilde_bs[ii][jj,:]
		zderivs[cur,:] *= psitilde_cs[ii][jj][kk,:]
		zderivs[cur,:] *= 0.5 * (1.0+eta1s[:])
		if ii>0:
		    zderivs[cur,:] *= eta2_scalings[ii-1]
		if ii+jj>0:
		    zderivs[cur,:] *= eta3_scalings[ii+jj-1]

		#tmp will hold d/deta2 term
		tmp[:] = psitilde_bs_derivs[ii][jj,:]
		tmp[:] *= eta2_scalings[ii,:]
		if ii>0:
		    tmp[:] -= 0.5*ii*eta2_scalings[ii-1]*psitilde_bs[ii][jj,:]
		tmp[:] *= psitilde_as[ii,:]
		tmp[:] *= psitilde_cs[ii][jj][kk,:]
		tmp[:] *= 0.5*(1.0+eta2s[:])
		if ii+jj>0:
		    tmp[:] *= eta3_scalings[ii+jj-1]
		zderivs[cur,:] += tmp[:]		 

		#tmp will hold d/deta3 term
		tmp[:] = psitilde_cs_derivs[ii][jj][kk,:]
		tmp[:] *= eta3_scalings[ii+jj,:]
		if ii+jj>0:
		    tmp[:] -= 0.5*(ii+jj)*psitilde_cs[ii][jj][kk,:] \
			* eta3_scalings[ii+jj-1,:]
		tmp[:] *= psitilde_as[ii,:]
		tmp[:] *= psitilde_bs[ii][jj,:]
		tmp[:] *= eta2_scalings[ii]
		zderivs[cur,:] += tmp[:]

                xderivs[cur,:] *= scale*math.sqrt( (ii+0.5) \
					     * (ii+jj+1.0) \
					     * (ii+jj+kk+1.5) )

                yderivs[cur,:] *= scale*math.sqrt( (ii+0.5) \
					     * (ii+jj+1.0) \
					     * (ii+jj+kk+1.5) )

                zderivs[cur,:] *= scale*math.sqrt( (ii+0.5) \
					     * (ii+jj+1.0) \
					     * (ii+jj+kk+1.5) )

                
		cur += 1

    return (xderivs,yderivs,zderivs)


tabulators = { shapes.LINE : tabulate_phis_line , \
	       shapes.TRIANGLE : tabulate_phis_triangle , \
	       shapes.TETRAHEDRON : tabulate_phis_tetrahedron }

deriv_tabulators = { shapes.LINE : tabulate_phis_derivs_line , \
		     shapes.TRIANGLE : tabulate_phis_derivs_triangle , \
		     shapes.TETRAHEDRON : tabulate_phis_derivs_tetrahedron }


#####################################################3
#################  OLD STUFF:

def tabulate_phis_derivs_tetrahedron_old(n,xs):
    """Tabulates all the derivatives of basis functions over the tetrahedron
    up to degree n at points xs"""
    # unpack coordinates
    etas = map( eta_tetrahedron , xs )
    eta1s = numpy.array( [ eta1 for (eta1,eta2,eta3) in etas ] )
    eta2s = numpy.array( [ eta2 for (eta1,eta2,eta3) in etas ] )
    eta3s = numpy.array( [ eta3 for (eta1,eta2,eta3) in etas ] )

    psitilde_as = jacobi.eval_jacobi_batch(0,0,n,eta1s)
    psitilde_as_derivs = jacobi.eval_jacobi_deriv_batch(0,0,n,eta1s)
    psitilde_bs = [ jacobi.eval_jacobi_batch(2*i+1,0,n-i,eta2s) \
		    for i in range(0,n+1) ]
    psitilde_bs_derivs = [ jacobi.eval_jacobi_deriv_batch(2*i+1,0,n-i,eta2s) \
                           for i in range(0,n+1) ]
    psitilde_cs = [ [ jacobi.eval_jacobi_batch(2*(i+j+1),0,\
					       n-i-j,eta3s) \
		      for j in range(0,n+1-i) ] for i in range(0,n+1) ]
    psitilde_cs_derivs = [ [ jacobi.eval_jacobi_deriv_batch(2*(i+j+1),0,\
                                                            n-i-j,eta3s) \
                             for j in range(0,n+1-i) ] for i in range(0,n+1) ]
    
    eta2_scalings = numpy.zeros( (n+1,len(xs)) ,"d")
    eta2_scalings[0,:] = 1.0
    if n > 0:
	eta2_scalings[1,:] = 0.5 * (1.0 - eta2s)
	for k in range(2,n+1):
	    eta2_scalings[k,:] = eta2_scalings[k-1,:] * eta2_scalings[1,:]

    eta3_scalings = numpy.zeros( (n+1,len(xs)) ,"d" )
    eta3_scalings[0,:] = 1.0
    if n > 0:
	eta3_scalings[1,:] = 0.5 * (1.0 - eta3s)
	for k in range(2,n+1):
	    eta3_scalings[k,:] = eta3_scalings[k-1,:] \
		* eta3_scalings[1,:]

    tmp = numpy.zeros( (len(xs),) , "d" )
    xderivs = numpy.zeros( (shapes.poly_dims[shapes.TETRAHEDRON](n), \
			      len(xs)) , \
			     "d" )
    yderivs = numpy.zeros( xderivs.shape , "d" )
    zderivs = numpy.zeros( xderivs.shape , "d" )

    cur = 0
    for k in range(0,n+1):  # loop over degree
	for i in range(0,k+1):
	    for j in range(0,k-i+1):
		ii = k-i-j
		jj = j
		kk = i

                # xderivative
                xderivs[cur,:] = psitilde_as_derivs[ii,:] \
                                 * psitilde_bs[ii][jj,:] \
                                 * psitilde_cs[ii][jj][kk,:]
                if ii > 0:
                    xderivs[cur,:] *= eta2_scalings[ii-1,:]
                if ii+jj>0:
                    xderivs[cur,:] *= eta3_scalings[ii+jj-1,:]

                # yderivative
                # eta1 derivative term
                yderivs[cur,:] = 0.5 * (1.0+eta1s) * xderivs[cur,:]

                # eta2 derivative term, start with the "internal"
                # product rule on the eta2 term itself
                tmp[:] = psitilde_bs_derivs[ii][jj,:] * eta2_scalings[ii,:]
                if ii>0:
                    tmp[:] -= 0.5 * ii * psitilde_bs[ii][jj,:] \
                              * eta2_scalings[ii-1,:]
                tmp[:] *= psitilde_as[ii,:] * psitilde_cs[ii][jj][kk,:]
                if ii+jj>0:
                    tmp[:] *= eta3_scalings[ii+jj-1,:]
                yderivs[cur,:] += tmp

                # zderivative
                # eta1 derivative term
                zderivs[cur,:] = 0.5 * (1.0+eta1s) * xderivs[cur,:]

		# check this term here...
                # eta2 derivative term
                zderivs[cur,:] += tmp * 0.5 * (1.0+eta1s)

		# and this one.
                # eta3 derivative term
                tmp[:] = psitilde_cs_derivs[ii][jj][kk,:] * eta3_scalings[ii+jj,:]
                if ii+jj>0:
                    tmp[:] += psitilde_cs[ii][jj][kk,:] * (ii+jj) * eta3_scalings[ii+jj-1,:]
                tmp[:] *= psitilde_as[ii,:] * psitilde_bs[ii][jj,:] \
                          * eta2_scalings[ii,:]
                zderivs[cur,:] += tmp

                xderivs[cur,:] *= math.sqrt( (ii+0.5) \
					     * (ii+jj+1.0) \
					     * (ii+jj+kk+1.5) )

                yderivs[cur,:] *= math.sqrt( (ii+0.5) \
					     * (ii+jj+1.0) \
					     * (ii+jj+kk+1.5) )

                zderivs[cur,:] *= math.sqrt( (ii+0.5) \
					     * (ii+jj+1.0) \
					     * (ii+jj+kk+1.5) )

                
		cur += 1

    return (xderivs,yderivs,zderivs)
