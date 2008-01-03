import numpy, polynomial,shapes, numpy.linalg, quadrature

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

def transform_dmats( A , dmats ):
    return [ numpy.array( reduce( lambda a,b: a+b ,
                                  [ A[j,i] * dmats[j] \
                                    for j in range( len(dmats) ) ] ) ) \
             for i in range( len( dmats ) ) ]
    
class AffineTransformedFunctionSpace:
    def __init__( self , fspace , verts ):
        (self.A,self.b) = pullback_mapping( verts )
        self.pullback = pullback_function( self.A , self.b )
        self.pushforward = pushforward_function(self.A,self.b)
        self.fspace = fspace
        self.verts = verts
        self.dmats = []
        self.dmats = transform_dmats( self.A , fspace.base.dmats )
#        for i in range(self.spatial_dimension()):
#            Acol = self.A[:,i]
#            self.dmats.extend( numpy.array( [ Acol[j] * fspace.base.dmats[j] \
#                               for j in range(self.spatial_dimension()) ] ) )
        return

    def degree( self ): return self.fspace.degree()
    def spatial_dimension( self ): return self.fspace.spatial_dimension()
    def __len__( self ): return self.fspace.__len__()

    def get_dmats( self ): return self.dmats

    def eval_all( self , x ):
        return self.tabulate( ( x , ) )[:,0]
    
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

    def rank( self ): return self.fspace.rank()
    def __len__( self ): return len( self.fspace )
    def __getitem__( self , i ):
        if type(i) == type(1): # single item
            newdof = numpy.zeros( self.fspace.coeffs[0].shape , "d" )
            newdof[i] = 1.0
            return AffineTransformedScalarPolynomial( self , newdof )
        else:
            return AffineTransformedFunctionSpace( self.fspace[i] , \
                                                   self.verts )

class PiolaTransformedFunctionSpace:
    def __init__( self , fspace , verts , div_or_curl ):
        self.fspace = fspace
        self.verts = verts
        (self.A,self.b) = pullback_mapping( verts )
        self.J = numpy.linalg.det( self.A )
        self.pullback = pullback_function( self.A , self.b )
        self.pushforward = pushforward_function(self.A,self.b)
        self.div_or_curl = div_or_curl

        # now need to make Piola
        if div_or_curl == "div":
            self.piola = lambda x: numpy.dot( self.A , x ) / self.J
        elif div_or_curl == "curl":
            self.Atrans = numpy.transpose( self.A , (1,0) )
            self.piola = lambda x: numpy.dot( self.Atrans , x )

        # make transformed coefficients
        cold = numpy.transpose( fspace.coeffs , (0 , 2 , 1) )
        cnewa = numpy.array( [ [ self.piola( c ) for c in coldrow ] \
                               for coldrow in cold ] )
        self.coeffs = numpy.transpose( cnewa , (0 ,2 ,1 ) )

        # make derivative matrices    
        self.dmats = transform_dmats( self.A , fspace.base.dmats )
#        self.dmats = [ numpy.array( [ self.A[j,i] * fspace.base.dmats[j] \
#                                      for j in range( self.spatial_dimension() ) ] ) \
#                       for i in range( self.spatial_dimension() ) ]
        return



    def degree( self ): return self.fspace.degree()
    def spatial_dimension( self ): return self.fspace.spatial_dimension()
    def __len__( self ): return self.fspace.__len__()
    def get_dmats( self ): return self.dmats
        
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
        

    def tensor_shape( self ):
        return self.fspace.tensor_shape( )

    def tabulate_jet( self , order , xs ):
        newxs = tuple( [ tuple( self.pullback( x ) ) for x in xs ] )
        return [ self.select_vector_component( i ).tabulate_jet( order , \
                                                                 newxs ) \
                 for i in range(self.tensor_shape()[0]) ]

    def rank( self ): return self.fspace.rank()
    def __getitem__( self , i ):
        if type(i) == type(1):
            return PiolaTransformedVectorPolynomial( self , self.coeffs[i] )
        else:
            return PiolaTransformedFunctionSpace( self.fspace[i] , \
                                                  self.verts , \
                                                  self.div_or_curl )

class AffineTransformedScalarPolynomial:
    def __init__( self , atfspace , dof ):
        # dof is the dof on the untransformed space
        self.atfspace, self.dof = atfspace, dof
        return
    def __call__( self , x ):
        vals = self.atfspace.eval_all( x )
        return numpy.dot( vals , self.dof )
    def deriv( self , i ):
        nspace = self.atfspace.deriv_all( i )
        return AffineTransformedScalarPolynomial( nspace , self.dof )
        

class PiolaTransformedVectorPolynomial:
    def __init__( self , ptfspace , dof ):
        self.ptfspace, self.dof = ptfspace, dof
        return
    def __call__( self , x ):
        newx = self.ptfspace.pullback( x )
        return numpy.dot( self.dof , self.ptfspace.fspace.base.eval_all( newx ) )
    def __getitem__( self , i ):
        nspace = self.ptfspace.select_vector_component(i)
        return AffineTransformedScalarPolynomial( nspace , self.dof )

class AffineTransformedQuadratureRule(quadrature.QuadratureRule):
    def __init__( self , q , verts ):
        self.q, self.verts = q,verts
        A,b = pullback_mapping( verts )
        pullback = pullback_function( A , b )
        newpts = map( pullback , q.get_points() )
        newwts = q.get_weights()/ numpy.linalg.det( A )
        quadrature.QuadratureRule.__init__( newpts , newwts )
        return

