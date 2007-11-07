import numpy, polynomial,shapes, numpy.linalg

# this works on simplices of arbitrary dimension
def pullback_mapping( verts ):
    spacedim = len(verts)-1
    mat = numpy.zeros( (spacedim*(spacedim+1),spacedim*(spacedim+1)) ,"d")
    rhs = numpy.zeros( (spacedim*(spacedim+1),1) , "d" )
    
    vertsmat = numpy.array( verts )

    refvertsdict = shapes.vertices[spacedim]
    refvertsvec = numpy.zeros( (spacedim*(spacedim+1),),"d")
    for i in range(spacedim+1):
        for j in range(spacedim):
            refvertsvec[i*spacedim+j] = refvertsdict[i][j]
    
    for k in range(spacedim+1):
        for j in range(spacedim):
            mat[spacedim*k+j,j*spacedim:(j+1)*spacedim] = vertsmat[k,:]
            mat[spacedim*k+j,spacedim**2+j] = 1.0
            
    sol = numpy.linalg.solve(mat,refvertsvec)

    A = numpy.reshape( sol[:spacedim**2] , (spacedim,spacedim))
    b = sol[spacedim**2:]

    return (A,b)

def pullback_function( A , b ):
    return lambda x:numpy.dot(A,x)+b

def pushforward_function(A,b):
    return lambda xhat:numpy.linalg.solve(A,xhat-b)

    
class AffineTransformedFunctionSpace:
    def __init__( self , fspace , verts ):
        (self.A,self.b) = pullback_mapping( verts )
        self.pullback = pullback_function( self.A , self.b )
        self.pushforward = pushforward_function(self.A,self.b)
        self.fspace = fspace
        self.verts = verts
        return

    def degree( self ): return self.fspace.degree()
    def spatial_dimension( self ): return self.fspace.spatial_dimension()
    def __len__( self ): return self.fspace.__len__()

    def eval_all( self , x ):
        xnew = self.pullback( x )
        return self.tabulate( numpy.array([xnew]))[:,0]
    
    def tabulate( self , pts ):
        newpts = tuple( [ tuple(self.pullback( x )) for x in pts ] )
        return self.fspace.tabulate( newpts )

    def deriv_all( self , i ):
        # first, get coefficients of derivatives in each direction
        Uhatprimecoeffs=[self.fspace.deriv_all( j ).coeffs\
                         for j in range(self.spatial_dimension())]
        Acol = self.A[:,i]
        newcoeffs = sum( numpy.array( [ Acol[j] * Uhatprimecoeffs[j]\
                      for j in range(self.spatial_dimension())] ) )
        Unew = polynomial.PolynomialSet( self.fspace.base , newcoeffs )
        return AffineTransformedFunctionSpace( Unew , self.verts )

    def multi_deriv_all( self , alpha ):
        U=self
        for c in range(len(alpha)):
            for i in range(alpha[c]):
                U=U.deriv_all(c)
        return U

    def tabulate_jet( self , order , xs ):
        alphas = [polynomial.mis(shapes.dimension(self.fspace.base.shape),i) \
                  for i in range(order+1) ]
        a = [None]*len(alphas)
        for i in range(len(alphas)):
            a[i]={}
            for alpha in alphas[i]:
                a[i][alpha]=self.multi_deriv_all(alpha).tabulate(xs)
        return a

    def trace_tabulate_jet( self , d , e , order , xs , drefverts ):
        # turn drefverts into FIAT's ref verts of the same dimension
        # this is a pull-back for the lower-dimensional space
        (Alow,blow) = pullback_mapping( drefverts )
        lowdimpullback = pullback_function(Alow,blow)

        xspullback = tuple( map( tuple , map( lowdimpullback , xs ) ) )

        # embed the pulled-back vertices into the right space
        spacedim = self.spatial_dimension()
        xsfulldimpullback = \
           map( shapes.pt_maps[spacedim][d](e),\
                xspullback )  
        
        # push them forward into the right space
        xs_on_cell = tuple( map( tuple , \
                                 map( self.pushforward ,\
                                      xsfulldimpullback ) ) )

        return self.tabulate_jet( order , xs_on_cell )

    def tensor_shape( self ):
        return (1,)


class HDivPiolaTransformedSpace:
    def __init__( self , fspace , verts ):
        self.fspace = fspace
        self.verts = verts
        (self.A,self.b) = pullback_mapping( verts )
        self.J = numpy.linalg.det( self.A )
        self.pullback = pullback_function( self.A , self.b )
        self.pushforward = pushforward_function(self.A,self.b)

        # now need to make Piola
        self.piola = lambda x: numpy.dot( self.A , x ) / self.J

        # make transformed coefficients
        cold = numpy.transpose( fspace.coeffs , (0 , 2 , 1) )
        cnewa = numpy.array( [ [ self.piola( c ) for c in coldrow ] \
                               for coldrow in cold ] )
        self.coeffs = numpy.transpose( cnewa , (0 ,2 ,1 ) )
        
    def eval_all( self , x ):
        """Returns arr[i,j] where i runs over the members of the
        set and j runs over the components of each member."""
        bvals = self.fspace.base.eval_all( self.pullback( x ) )
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
        newxs = tuple( [ tuple( self.pullback(x))  for x in xs] )
        bvals = self.fspace.base.tabulate( newxs ) 
        old_shape = self.coeffs.shape
        flat_coeffs = numpy.reshape( self.coeffs , \
                                       ( old_shape[0]*old_shape[1] , \
                                         old_shape[2] ) )
        flat_dot = numpy.dot( flat_coeffs , bvals )
        unflat_dot = numpy.reshape( flat_dot , \
                                ( old_shape[0] , old_shape[1] , len(xs) ) )
        return unflat_dot

    def select_vector_component( self , i ):
        newfs = polynomial.ScalarPolynomialSet( self.fspace.base ,
                                               self.coeffs[:,i,:] )
        return AffineTransformedFunctionSpace( newfs , self.verts )

    def trace_tabulate_jet( self , d , e , order , xs , drefverts ):
        (Alow,blow) = pullback_mapping( drefverts )
        lowdimpullback = pullback_function(Alow,blow)

        xspullback = tuple( map( tuple , map( lowdimpullback , xs ) ) )

        # embed the pulled-back vertices into the right space
        spacedim = self.spatial_dimension()
        xsfulldimpullback = \
           map( shapes.pt_maps[spacedim][d](e),\
                xspullback )  
        
        # push them forward into the right space
        xs_dim = tuple( map( tuple , \
                             map( self.pushforward ,\
                                  xsfulldimpullback ) ) )

        return [ self.select_vector_component( i ).tabulate_jet( order , \
                                                                 xs_dim ) \
                 for i in range(self.tensor_shape()[0]) ]
        

        return

    def tensor_shape( self ):
        return self.fspace.tensor_shape( )

    def tabulate_jet( self , order , xs ):
        newxs = tuple( [ tuple( self.pullback( x ) ) for x in xs ] )
        return [ self.select_vector_component( i ).tabulate_jet( order , \
                                                                 newxs ) \
                 for i in range(self.tensor_shape()[0]) ]

