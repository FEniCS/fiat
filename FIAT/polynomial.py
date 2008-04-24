# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# modified 2 Aug 2005 by RCK to add "trace_tabulate" method
# modified 1 June 2005 by RCK


import expansions, shapes, numpy, numpy.linalg
from numpy.linalg import svd

class AbstractPolynomialBase( object ):
	# I need to define the eval_all, tabulate, degree,
	# spatial_dimension, domain_shape, __len__, __getitem__
	# functions, plus the dmats
	pass

#class MonomialPolynomialBase( AbstractPolynomialBase ):
#	def __init__( self , shape , n ):
		

class OrthonormalPolynomialBase( AbstractPolynomialBase ):
	"""Defines an object that can tabulate the orthonormal polynomials
	of a particular degree on a particular shape """
	def __init__( self , shape , n ):
		"""shape is a shape code defined in shapes.py	
		n is the polynomial degree"""
		self.bs = expansions.make_expansion( shape , n )
		self.shape,self.n = shape,n
		d = shapes.dimension( shape )
		if n == 0:
			self.dmats = [ numpy.array( [ [ 0.0 ] ] , "d" ) ] * d
		else:
			dtildes = [ numpy.zeros( (len(self.bs),len(self.bs)) , "d" ) \
                      for i in range(0,d) ]
			pts = shapes.make_points( shape , d , 0 , n + d + 1 )
			v = numpy.transpose( expansions.tabulators[shape](n,pts) )
			vinv = numpy.linalg.inv( v )
			dtildes = [numpy.transpose(a) \
		       for a in expansions.deriv_tabulators[shape](n,pts)]
			self.dmats = [ numpy.dot( vinv , dtilde ) for dtilde in dtildes ]
		return

	def eval_all( self , x ):
		"""Returns the array A[i] = phi_i(x)."""
		return self.tabulate( numpy.array([x]))[:,0]

	def tabulate( self , xs ):
		"""xs is an iterable object of points.
		returns an array A[i,j] where i runs over the basis functions
		and j runs over the elements of xs."""
		return expansions.tabulators[self.shape](self.n,numpy.array(xs))

	def degree( self ):
		return self.n

	def spatial_dimension( self ):
		return shapes.dimension( self.shape )
	def domain_shape( self ):
		return self.shape 

	def __len__( self ):
		"""Returns the number of items in the set."""
		return len( self.bs )

	def __getitem__( self , i ):
		return self.bs[i]


class AbstractPolynomialSet( object ):
    """A PolynomialSet is a collection of polynomials defined as
    linear combinations of a particular base set of polynomials.
    In this code, these are the orthonormal polynomials defined
    on simplices in one, two, or three spatial dimensions.
    PolynomialSets may contain array-valued elements."""
    def __init__( self , base , coeffs ):
        """Base is an instance of PolynomialBase.
        coeffs[i][j] is the coefficient (possibly array-valued)
        of the j:th orthonormal function in the i:th member of
        the polynomial set.  If rank("""
        self.base, self.coeffs = base, coeffs
        return
    def __getitem__( self , i ):
        """If i is an integer, returns the i:th member of the set
        as the appropriate subtype of AbstractPolynomial.  If
        i is a slice, returns the the implied subset of polynomials
        as the appropriate subtype of AbstractPolynomialSet."""
        if type(i) == type(1):  # single item
            return Polynomial( self.base , self.coeffs[i] )
        elif type(i) == type(slice(1)):
            return PolynomialSet( self.base , self.coeffs[i] )
    def eval_all( self , x ):
        """Returns the array A[i] of the i:th member of the set
        at point x."""
        pass
    def tabulate( self , xs ):
        """Returns the array A[i][j] with the i:th member of the
        set evaluated at the j:th member of xs."""
        pass
    def trace_tabulate( self , d , e , xs ):
        """Tabulates the basis on entity e of topological dimension
        d at points xs, which are specified on the reference element
        of dimension d."""
        myshape = self.domain_shape()
        xs_dim = map( shapes.pt_maps[ myshape ][ d ]( e ) , xs )
        return self.tabulate( xs_dim )
    def degree( self ):
        """Returns the polynomial degree of the space.  If the polynomial
        set lies between two degrees, such as the Raviart-Thomas space, then
        the degree of the smallest space of complete degree containing the
        set is returned.  For example, the degree of RT0 is 1 since some
        linear functions are required to represent the basis."""
        return self.base.degree()
    def rank( self ):
        """Returns the tensor rank of the range of the functions (0
        for scalar, two for vector, etc"""
        return numpy.rank( self.coeffs ) - 2
    def tensor_dim( self ):
        if self.rank() == 0:
            return tuple()
        else:
            return self.coeffs.shape[1:-1]
    def tensor_shape( self ):
        pass
    def spatial_dimension( self ):
        return shapes.dimension( self.base.shape )
    def domain_shape( self ):
        """Returns the code for the element shape of the domain.  This
        is the code given in module shapes.py"""
        return self.base.shape
    def __len__( self ):
        """Returns the number of items in the set."""
        return len( self.coeffs )
    def take( self , items ):
        """Extracts the subset of polynomials given by indices stored
        in iterable items.""" 
        return PolynomialSet( self.base , numpy.take( self.coeffs , items , 0 ) )

# ScalarPolynomialSet can be made with either
# -- ScalarPolynomialSet OR
# -- PolynomialBase
# along with a rank 2 array, where each row is the
# coefficients of the members of the base for
# a member of the new set
class ScalarPolynomialSet( AbstractPolynomialSet ):
    def __init__( self , base , coeffs ):
        if numpy.rank( coeffs ) != 2:
            raise RuntimeError, \
                  "Illegal coeff matrix: ScalarPolynomialSet"
        if isinstance( base, AbstractPolynomialBase ):
            AbstractPolynomialSet.__init__( self , base , coeffs )
        elif isinstance( base , ScalarPolynomialSet ):
            new_base = base.base
            new_coeffs = numpy.dot( coeffs , base.coeffs )
            AbstractPolynomialSet.__init__( self , new_base , new_coeffs )
        return

    def select_vector_component( self , i ):
        if i != 0: raise RuntimeError, "Illegal indexing into ScalarPolynomialSet"
        return self

    def eval_all( self , x ):
        """Returns the array A[i] = psi_i(x)."""
        bvals = self.base.eval_all( x )
        return numpy.dot( self.coeffs , bvals )

    def tabulate( self , xs ):
        """Returns the array A[i][j] with i running members of the set
        and j running over the members of xs."""
        bvals = self.base.tabulate( xs )
        return numpy.dot( self.coeffs , bvals )    

    def deriv_all( self , i ):
        """Returns the PolynomialSet containing the partial derivative
        in the i:th direction of each component."""
        D = self.base.dmats[i]
        new_coeffs = numpy.dot( self.coeffs , numpy.transpose(D) )
        return ScalarPolynomialSet( self.base , new_coeffs )
    
    def multi_deriv_all( self , alpha ):
        """Returns the PolynomialSet containing the alpha partial
        derivative of everything.  alpha is a multiindex of the same
        size as the spatial dimension."""
        U = self
        for c in range(len(alpha)):
            for i in range(alpha[c]):
                U = U.deriv_all( c )
        return U
    
    def tabulate_jet( self , order , xs ):
        """Computes all partial derivatives of the members of the set
        up to order.  Returns an array of dictionaries
        a[i][mi] where i is the order of partial differentiation
        and mi is a multiindex with |mi| = i.  The value of
        a[i][mi] is an array A[i][j] containing the appropriate derivative
        of the i:th member of the set at the j:th member of xs."""
	
        alphas = [ mis( shapes.dimension( self.base.shape ) , i ) \
                   for i in range(order+1) ]

        a = [ None ] * len(alphas)
        for i in range(len(alphas)):
            a[i] = {}
            for alpha in alphas[i]:
                a[i][alpha] = self.multi_deriv_all( alpha ).tabulate( xs )
        return a

    def trace_tabulate_jet( self , d , e , order, xs ):
        """Computes all partial derivatives of the members of the set
        up to order on entity e of topological dimension d at points
        xs, which are specified on the reference element of dimension d.
        Returns an array of dictionaries a[i][mi] where i is the order
        of partial differentiation and mi is a multiindex with |mi| = i.
        The value of a[i][mi] is an array A[i][j] containing the
        appropriate derivative of the i:th member of the set at the
        j:th member of xs."""
        myshape = self.domain_shape()
        xs_dim = map( shapes.pt_maps[ myshape ][ d ]( e ) , xs )
        return self.tabulate_jet( order, xs_dim )


    def tensor_shape( self ):
        return (1,)

# A VectorPolynomialSet may be made with either
# -- PolynomialBase and rank 3 array of coefficients OR
# -- VectorPolynomialSet and rank 2 array

# coeffs for a VectorPolynomialSet is an array
# C[i,j,k] where i runs over the members of the set,
# j runs over the components of the vectors, and
# k runs over the members of the base set.

class VectorPolynomialSet( AbstractPolynomialSet ):
    """class modeling sets of vector-valued polynomials."""
    def __init__( self , base , coeffs ):
        if isinstance( base , AbstractPolynomialBase ):
            if numpy.rank( coeffs ) != 3:
                raise RuntimeError, \
                      "Illegal coeff matrix: VectorPolynomialSet"
            AbstractPolynomialSet.__init__( self , base , coeffs )
            pass
        elif isinstance( base , VectorPolynomialSet ):
            if numpy.rank( coeffs ) != 2:
                raise RuntimeError, \
                      "Illegal coeff matrix: VectorPolynomialSet"
            new_base_shape = (base.coeffs.shape[0], \
                         reduce(lambda a,b:a*b, \
                                base.coeffs.shape[1:] ) )
            base_coeffs_reshaped = numpy.reshape( base.coeffs , \
                                                    new_base_shape )
            new_coeffs_shape = tuple( [coeffs.shape[0]] \
                                      + list(base.coeffs.shape[1:]) )
            new_coeffs_flat = numpy.dot( coeffs , base_coeffs_reshaped )
            new_coeffs = numpy.reshape( new_coeffs_flat , new_coeffs_shape )
            AbstractPolynomialSet.__init__( self , base.base , new_coeffs )
            return
        else:
            raise RuntimeError, "Illegal base set: VectorPolynomialSet"

    def eval_all( self , x ):
        """Returns the array A[i,j] where i runs over the members of the
        set and j runs over the components of each member."""
        bvals = self.base.eval_all( x )
        old_shape = self.coeffs.shape
        flat_coeffs = numpy.reshape( self.coeffs , \
                                       (old_shape[0]*old_shape[1] , \
                                        old_shape[2] ) )
        flat_dot = numpy.dot( flat_coeffs , bvals )
        return numpy.reshape( flat_dot , old_shape[:2] )
                      
    def tabulate( self , xs ):
	"""xs is an iterable object of points.
	returns an array A[i,j,k] where i runs over the members of the
	set, j runs over the components of the vectors, and k runs
	over the points."""
        bvals = self.base.tabulate( xs )
        old_shape = self.coeffs.shape
        flat_coeffs = numpy.reshape( self.coeffs , \
                                       ( old_shape[0]*old_shape[1] , \
                                         old_shape[2] ) )
        flat_dot = numpy.dot( flat_coeffs , bvals )
        unflat_dot = numpy.reshape( flat_dot , \
                                ( old_shape[0] , old_shape[1] , len(xs) ) )
        return unflat_dot

    def select_vector_component( self , i ):
        """Returns the ScalarPolynomialSetconsisting of the i:th
        component of allthe vectors.  It keeps zeros around.  This
        could change in later versions."""
        return ScalarPolynomialSet( self.base , self.coeffs[:,i,:] )

    def trace_tabulate_jet( self , d , e , order , xs ):
        myshape = self.domain_shape()
        xs_dim = map( shapes.pt_maps[ myshape ][ d ]( e ) , xs ) 
        return [ self.select_vector_component( i ).tabulate_jet( order , \
                                            xs_dim ) \
	            for i in range(self.tensor_shape()[0]) ]

    def tensor_shape( self ):
        return self.coeffs.shape[1:2]

    def tabulate_jet( self , order , xs ):
        return [ self.select_vector_component( i ).tabulate_jet( order , xs ) \
                 for i in range(self.tensor_shape()[0]) ]


# The code for TensorPolynomialSet will look just like VectorPolynomialSet
# except that we will flatten all of the coefficients to make them
# look like vectors instead of tensors.

class TensorPolynomialSet( AbstractPolynomialSet ):
    pass

def PolynomialSet( base , coeffs ):
    """Factory function that takes some PolynomialBase or
    AbstractPolynomialSet and a collection of coefficients and
    either returns the appropriate kind of subclass of
    AbstractPolynomialSet or else raises an exception."""
    if len( coeffs.shape ) == 2:
        if isinstance( base , AbstractPolynomialBase ):
            return ScalarPolynomialSet( base , coeffs )
        elif isinstance( base , VectorPolynomialSet ):
            return VectorPolynomialSet( base , coeffs )
        elif isinstance( base , ScalarPolynomialSet ):
            return ScalarPolynomialSet( base , coeffs )
        else:
            raise RuntimeError, "???: PolynomialSet"
    elif len( coeffs.shape ) == 3:
        return VectorPolynomialSet( base , coeffs )
    elif len( coeffs.shape ) > 3:
        return TensorPolynomialSet( base , coeffs )
    else:
        print coeffs
        raise RuntimeError, "Unknown error, PolynomialSet"                      


class AbstractPolynomial( object ):
    """Base class from which scalar- vector- and tensor- valued
    polynomials are implemented.  At least, all types of
    polynomials must support:
    -- evaluation via __call__().
    All polynomials are represented as a linear combination of
    some base set of polynomials."""
    def __init__( self , base , dof ):
        """Constructor for AbstractPolynomial -- must be provided
        with PolynomialBase instance and a numpy.array of
        coefficients."""
        if not isinstance( base , AbstractPolynomialBase ):
            raise RuntimeError, "Illegal base: AbstractPolynomial"
        self.base , self.dof = base , dof
        return
    def __call__( self , x ):    pass
    def __getitem__( self , x ): pass
    def degree( self ): return self.base.degree()

class ScalarPolynomial( AbstractPolynomial ):
    """class of scalar-valued polynomials supporting
    evaluation and differentiation."""
    def __init__( self , base , dof ):
        if numpy.rank( dof ) != 1:
            raise RuntimeError, \
                  "This isn't a scalar polynomial."
        AbstractPolynomial.__init__( self , base , dof )
        return
    def __call__( self , x ):
        """Evaluates the polynomial at point x"""
        bvals = self.base.eval_all( x )
        return numpy.dot( self.dof , bvals )
    def deriv( self , i ):
        """Computes the partial derivative in the i:th direction,
        represented as a polynomial over the same base."""
        b = self.base
        D = b.dmats[i]
        deriv_dof = numpy.dot( D , self.dof )
        return ScalarPolynomial( b , deriv_dof )
    def __getitem__( self , i ):
        if i != 0: raise RuntimeError, "Illegal indexing into ScalarPolynomial"
        return self

class VectorPolynomial( AbstractPolynomial ):
    """class of vector-valued polynomials supporting evaluation
    and component selection."""
    def __init__( self , base , dof ):
        if numpy.rank( dof ) != 2 :
            raise RuntimeError, "This isn't a vector polynomial."
        AbstractPolynomial.__init__( self , base , dof )
        return
    def __call__( self , x ):
        bvals = self.base.eval_all( x )
        return numpy.dot( self.dof , bvals )
    def __getitem__( self , i ):
        if type( i ) != type( 1 ):
            raise RuntimeError, "Illegal input type."
        return ScalarPolynomial( self.base , self.dof[i] )

class TensorPolynomial( AbstractPolynomial ):
    pass

# factory function that determines whether to instantiate
# a scalar, vector, or general tensor polynomial
def Polynomial( base , dof ):
    """Returns a instance of the appropriate subclass of
    AbstractPolynomial."""
    if not isinstance( base , AbstractPolynomialBase ) and \
       not isinstance( base , PolynomialSet ):
        raise RuntimeError, "Illegal types, Polynomial"
    if numpy.rank( dof ) == 1:
        return ScalarPolynomial( base , dof )
    elif numpy.rank( dof ) == 2:
        return VectorPolynomial( base , dof )
    elif numpy.rank( dof ) > 2:
        return TensorPolynomial( base , dof )
    else:
        raise RuntimeError, "Illegal shape dimensions"

def OrthogonalPolynomialSet( element_shape , degree ):
    """Returns a ScalarPolynomialSet that models the
    orthormal basis functions on element_shape.  This allows
    us to arbitrarily differentiate the orthogonal polynomials
    if we need to."""
    b = OrthonormalPolynomialBase( element_shape , degree )
    coeffs = numpy.identity( shapes.poly_dims[element_shape](degree) , \
                               "d" )
    return ScalarPolynomialSet( b , coeffs )

def OrthogonalPolynomialArraySet( element_shape , degree , nc = None ):
    """Returns a VectorPolynomialSet that models the orthnormal basis
    for vector-valued functions with d components, where d is the spatial
    dimension of element_shape"""
    b = OrthonormalPolynomialBase( element_shape , degree )
    space_dim = shapes.dimension( element_shape )
    if nc == None:
        nc = space_dim
    M = shapes.polynomial_dimension( element_shape , degree )
    coeffs = numpy.zeros( (nc * M , nc , M ) , "d" )
    ident = numpy.identity( M , "d" )
    for i in range(nc):
        coeffs[(i*M):(i+1)*M,i,:] = ident[:,:]

    return PolynomialSet( b , coeffs )

class FiniteElement( object ):
    """class modeling the Ciarlet abstraction of a finite element"""
    def __init__( self , Udual , U ):
        self.Udual = Udual
        v = outer_product( Udual.get_functional_set() , U )
        vinv = numpy.linalg.inv( v )
        self.U = PolynomialSet( U , numpy.transpose(vinv) )
        return
    def domain_shape( self ): return self.U.domain_shape()
    def function_space( self ): return self.U
    def dual_basis( self ): return self.Udual

def ConstrainedPolynomialSet( fset ):
    # takes a FunctionalSet object and constructs the PolynomialSet
    # consisting of the intersection of the null spaces of its member
    # functionals acting on its function set
    tol = 1.e-12
    L = outer_product( fset , fset.function_space() )
    (U_L , Sig_L , Vt_L) = svd( L , 1 ) # full svd

    # some of the constraint functionals may be redundant.  I
    # must check to see how many singular values there are, as this
    # is what determines the rank and nullity of L
    Sig_L_nonzero = numpy.array( [ a for a in Sig_L if abs(a) > tol ] )
    num_nonzero_svs = len( Sig_L_nonzero )
    vcal = Vt_L[ num_nonzero_svs: ]
    return PolynomialSet( fset.function_space() , vcal )

def outer_product(flist,plist):
    # if the functions are vector-valued, I need to reshape the coefficient
    # arrays so they are two-dimensional to do the dot product
    shp = flist.mat.shape
    if len(shp) > 2:
        num_cols = reduce( lambda a,b:a*b , shp[1:] )
        new_shp = (shp[0],num_cols)
        A = numpy.reshape( flist.mat,(flist.mat.shape[0],num_cols) )
        B = numpy.reshape( plist.coeffs,(plist.coeffs.shape[0],num_cols) )
    else:
        A = flist.mat
        B = plist.coeffs

    return numpy.dot( A , numpy.transpose( B ) )
       
def gradient( u ):
    if not isinstance( u , ScalarPolynomial ):
        raise RuntimeError, "Illegal input to gradient"
    new_dof = numpy.zeros( (u.base.spatial_dimension() , len(u.dof) ) , "d" )
    for i in range(u.base.spatial_dimension()):
        new_dof[i,:] = numpy.dot( u.base.dmats[i] , \
                                    u.dof )
    return VectorPolynomial( u.base , new_dof )

def gradients( U ):
    """For U a ScalarPolynomialSet, computes the gradient of each member
    of U, returning a VectorPolynomialSet."""
    if not isinstance( U , ScalarPolynomialSet ):
        raise RuntimeError, "Illegal input to gradients"
    # new matrix is len(U) by spatial_dimension by length of poly base
    new_dofs = numpy.zeros( (len(U),U.spatial_dimension(),len(U.base)) , "d" )
    for i in range(U.spatial_dimension()):
        new_dofs[:,i,:] = numpy.dot( U.coeffs , U.base.dmats[i] )
    return VectorPolynomialSet( U.base , new_dofs )

def divergence( u ):
    if not isinstance( u , VectorPolynomial ):
        raise RuntimeError, "Illegal input to divergence"
    new_dof = numpy.zeros( len(u.base) , "d" )
    for i in range(u.base.spatial_dimension()):
        new_dof[:] += numpy.dot( u.base.dmats[i] , \
                                   u.dof[i] )
    return ScalarPolynomial( u.base , new_dof )

def curl( u ):
    if not isinstance( u , VectorPolynomial ):
        raise RuntimeError, "Illegal input to curl"
    if u.base.domain_shape() != shapes.TETRAHEDRON:
        raise RuntimeError, "Illegal shape to curl"
    new_dof = numpy.zeros( (3,len(u.base)) , "d" )
    new_dof[0] = u[2].deriv(1).dof - u[1].deriv(2).dof
    new_dof[1] = u[0].deriv(2).dof - u[2].deriv(0).dof
    new_dof[2] = u[1].deriv(0).dof - u[0].deriv(1).dof
    return VectorPolynomial( u.base, new_dof )

# |  i   j   k |
# | d0  d1  d2 |
# | v0  v1  v2 |
# -->
# (d1*v2-d2*v1,d2*v0-d0*v2,d0*v1-d1*v0)
# OR
# |  0 -d2  d1 | | v0 |
# | d2   0 -d0 | | v1 |
# | -d1 d0   0 | | v2 |
# SO...
# curl^t:
# |  0   d2t  -d1t | 
# | -d2t  0    d0t |
# |  d1t -d0t   0  |

def curl_transpose( u ):
    if not isinstance( u , VectorPolynomial ):
        raise RuntimeError, "Illegal input to curl"
    if u.base.domain_shape() != shapes.TETRAHEDRON:
        raise RuntimeError, "Illegal shape to curl"
    new_dof = numpy.zeros( (3,len(u.base)) , "d" )
    new_dof[0] = numpy.dot( numpy.transpose( u.base.dmats[2] ) , \
                              u[1].dof ) \
                 - numpy.dot( numpy.transpose( u.base.dmats[1] ) , \
                                u[2].dof )
    new_dof[1] = numpy.dot( numpy.transpose( u.base.dmats[0] ) , \
                              u[2].dof ) \
                 - numpy.dot( numpy.transpose( u.base.dmats[2]) , \
                                u[0].dof )
    new_dof[2] = numpy.dot( numpy.transpose( u.base.dmats[1] ) , \
                              u[0].dof ) \
                 - numpy.dot( numpy.transpose( u.base.dmats[0] ) , \
                                u[1].dof )
    return VectorPolynomial( u.base , new_dof )
                                 
    
# this is used in tabulating jets.        
def mis(m,n):
    """returns all m-tuples of nonnegative integers that sum up to n."""
    if m==1:
        return [(n,)]
    elif n==0:
        return [ tuple([0]*m) ]
    else:
        return [ tuple([n-i]+list(foo)) \
                 for i in range(n+1) \
                     for foo in mis(m-1,i) ]

# projection of some function onto a scalar polynomial set.

def projection( U , f , Q ):
    f_at_qps = [ f(x) for x in Q.get_points() ]
    phis_at_qps = U.tabulate( Q.get_points() )
    return ScalarPolynomial( U.base , \
                             numpy.array( [ sum( Q.get_weights() \
                                                   * f_at_qps \
                                                   * phi ) \
                                              for phi in phis_at_qps ] ) )

# C --> u,sig,vt.  first r columns of u span column range,
# so first r columns of v span row range
# so take first r rows of vt

def poly_set_union( U , V ):
    """Takes the union of two polynomial sets by appending their
    coefficient tensors and computing the range of the resulting set
    by the SVD."""
    new_coeffs = numpy.array( list( U.coeffs ) + list( V.coeffs ) )
    func_shape = new_coeffs.shape[1:]
    if len( func_shape ) == 1:
        (u,sig,vt) = svd( new_coeffs )
        num_sv = len( [ s for s in sig if abs( s ) > 1.e-10 ] )
        return PolynomialSet( U.base , vt[:num_sv] )
    else:
        new_shape0 = new_coeffs.shape[0]
        new_shape1 = reduce(lambda a,b:a*b,func_shape)
        nc = numpy.reshape( new_coeffs , (new_shape0,new_shape1) )
        (u,sig,vt) = svd( nc , 1 )
        num_sv = len( [ s for s in sig if abs( s ) > 1.e-10 ] )
        coeffs = vt[:num_sv]
        return PolynomialSet( U.base , \
                              numpy.reshape( coeffs , \
                                               tuple( [len(coeffs)] \
                                                      + list( func_shape ) ) ) )
