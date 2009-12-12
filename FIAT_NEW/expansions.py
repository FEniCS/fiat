# Copyright 2008 by Robert C. Kirby (Texas Tech University)
# License: LGPL
"""Principal orthgonal expansion functions as defined by Karniadakis
and Sherwin.  These are parametrized over a reference element so as
to allow users to get coordinates that they want."""

import reference_element
import numpy,math
import jacobi, reference_element

# Import AD modules from ScientificPython
import Scientific.Functions.Derivatives as Derivatives
import Scientific.Functions.FirstDerivatives as FirstDerivatives

def eta_triangle( xi ):
    """Maps from the (-1,1) reference triangle to [-1,1]^2."""
    (xi1,xi2) = xi
    if xi2 == 1.0:
        eta1 = -1.0
    else:
        eta1 = 2.0 * ( 1.0 + xi1 ) / (1.0 - xi2 ) - 1.0
    eta2 = xi2
    return eta1 , eta2    

def xi_triangle( eta ):
    """Maps from [-1,1]^2 to the (-1,1) reference triangle."""
    eta1,eta2 = eta
    xi1 = 0.5 * ( 1.0 + eta1 ) * ( 1.0 - eta2 ) - 1.0
    xi2 = eta2
    return (xi1,xi2)

def eta_tetrahedron( xi ):
    """Maps from the (-1,-1,-1) reference tet to [-1,1]^3"""
    xi1,xi2,xi3=xi
    if xi2+xi3 == 0.0:
        eta1 = 1.0
    else:
        eta1 = -2. * ( 1. + xi1 ) / (xi2 + xi3) - 1.
    if xi3 == 1.:
        eta2 = -1.
    else:
        eta2 = 2. * (1. + xi2) / (1. - xi3 ) - 1.
    eta3 = xi3

    return eta1,eta2,eta3

def xi_tetrahedron( eta ):
    """Maps from [-1,1]^3 to the -1/1 reference tetrahedron."""
    eta1,eta2,eta3 = eta
    xi1 = 0.25 * ( 1. + eta1 ) * ( 1. - eta2 ) * ( 1. - eta3 ) - 1.
    xi2 = 0.5 * ( 1. + eta2 ) * ( 1. - eta3 ) - 1.
    xi3 = eta3
    return xi1,xi2,xi3

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


def make_dmats( ref_el , n ):
    sd = ref_el.get_spatial_dimension( )
    if n == 0:
        return [ numpy.zeros( (1,1) , "d" ) ] * sd
    else:
        pts = ref_el.make_lattice( n )
        es = get_expansion_set( ref_el )
        v = numpy.transpose( es.tabulate( n , pts ) )
        vinv = numpy.linalg.inv( v )

        dpts = [ tuple( [FirstDerivatives.DerivVar( x[i] , i ) \
                             for i in range( len( x ) ) ] ) \
                     for x in pts ]

        dv = numpy.transpose( es.tabulate( n , dpts ) )
        
        
        
        dmats = []

        for i in range( sd ):
            dtilde_i = numpy.array( [ [ dvqr[1][i] for dvqr in dvrow ] \
                                          for dvrow in dv ] )
            dmats.append( numpy.dot( vinv , dtilde_i ) )

        return dmats
        

class LineExpansionSet:
    """Evaluates the Legendre basis on a line reference element."""
    def __init__( self , ref_el ):
        if ref_el.get_spatial_dimension() != 1:
            raise Exception, "Must have a line"
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultLine()
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A,self.b = reference_element.make_affine_mapping( v1 , v2 )       
        self.mapping = lambda x: numpy.dot( self.A , x ) + self.b
        self.scale = numpy.sqrt( numpy.linalg.det( self.A ) )

    def tabulate( self , n , pts ):
        """Returns a numpy array A[i,j] = phi_i( pts[j] )"""
        ref_pts = numpy.array([ self.mapping( pt ) for pt in pts ])
        psitilde_as = jacobi.eval_jacobi_batch(0,0,n,ref_pts)

        results = numpy.zeros( ( n+1 , len(pts) ) , type( pts[0][0] ) )
        for k in range( n + 1 ):
            results[k,:] = psitilde_as[k,:] * numpy.sqrt( k + 0.5 )

        return results        

    def tabulate_derivatives( self , n , pts ):
        """Returns a tuple of length one (A,) such that
        A[i,j] = D phi_i( pts[j] ).  The tuple is returned for
        compatibility with the interfaces of the triangle and
        tetrahedron expansions."""
        ref_pts = [ self.mapping( pt ) for pt in pts ]
        psitilde_as_derivs = jacobi.eval_jacobi_deriv_batch(0,0,n,ref_pts)

        results = numpy.zeros( ( n+1 , len(pts) ) , "d" )
        for k in range( 0 , n + 1 ):
            results[k,:] = psitilde_as_derivs[k,:] * numpy.sqrt( k + 0.5 )

        return (results,)
        

class TriangleExpansionSet:
    """Evaluates the orthonormal Dubiner basis on a triangular
    reference element."""
    def __init__( self , ref_el ):
        if ref_el.get_spatial_dimension() != 2:
            raise Exception, "Must have a triangle"
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultTriangle( )
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A,self.b = reference_element.make_affine_mapping( v1 , v2 )       
        self.mapping = lambda x: numpy.dot( self.A , x ) + self.b
#        self.scale = numpy.sqrt( numpy.linalg.det( self.A ) )
        
    def get_num_members( self , n ):
        return (n+1)*(n+2)/2

    def tabulate( self , n , pts ):
        if len( pts ) == 0:
            return numpy.array( [] )

        ref_pts = [ self.mapping( pt ) for pt in pts ]

        def idx(p,q):
            return (p+q)*(p+q+1)/2 + q

        def jrc( a , b , n ):
            an = float( ( 2*n+1+a+b)*(2*n+2+a+b)) \
                / float( 2*(n+1)*(n+1+a+b))
            bn = float( (a*a-b*b) * (2*n+1+a+b) ) \
                / float( 2*(n+1)*(2*n+a+b)*(n+1+a+b) )
            cn = float( (n+a)*(n+b)*(2*n+2+a+b)  ) \
                / float( (n+1)*(n+1+a+b)*(2*n+a+b) )
            return an,bn,cn


        pt_types = [ type(p) for p in pts[0] ]
        ntype = type(0.0)
        for pt in pt_types:
            if type(pt) != type(0.0):
                ntype = type(pt)
                break

        results = numpy.zeros( ( (n+1)*(n+2)/2,len(pts)),ntype )
        apts = numpy.array( pts )

        for ii in range( results.shape[1] ):
            results[0,ii] = 1.0 + apts[ii,0]-apts[ii,0]+apts[ii,1]-apts[ii,1]

        if n == 0:
            return results

        x = numpy.array( [ pt[0] for pt in ref_pts ] )
        y = numpy.array( [ pt[1] for pt in ref_pts ] )

        f1 = (1.0+2*x+y)/2.0
        f2 = (1.0 - y) / 2.0
        f3 = f2**2

        results[idx(1,0),:] = f1

        for p in range(1,n):
            a = (2.0*p+1)/(1.0+p)
            b = p / (p+1.0)
            results[idx(p+1,0)] = a * f1 * results[idx(p,0),:] \
                - p/(1.0+p) * f3 *results[idx(p-1,0),:]
        
        for p in range(n):
            results[idx(p,1),:] = 0.5 * (1+2.0*p+(3.0+2.0*p)*y) \
                * results[idx(p,0)]
            
        for p in range(n-1):
            for q in range(1,n-p):
                (a1,a2,a3) = jrc(2*p+1,0,q)
                results[idx(p,q+1),:] \
                    = ( a1 * y + a2 ) * results[idx(p,q)] \
                    - a3 * results[idx(p,q-1)]

        for p in range(n+1):
            for q in range(n-p+1):
                results[idx(p,q),:] *= math.sqrt((p+0.5)*(p+q+1.0))

        return results
        #return self.scale * results

    def tabulate_jet( self , n , pts , order = 1 ):
        import sys
        from Derivatives import DerivVar
        dpts = [ tuple( [ DerivVar( pt[i] , i , order ) \
                              for i in range( len( pt ) ) ] ) for pt in pts ]
        dbfs = self.tabulate( n , dpts )
        result = []
        for d in range( order + 1 ):
            result_d = [ [ foo[d] for foo in bar ] for bar in dbfs ]
            result.append( numpy.array( result_d ) )

        return result
                                       

class TetrahedronExpansionSet:
    """Collapsed orthonormal polynomial expanion on a tetrahedron."""
    def __init__( self , ref_el ):
        if ref_el.get_spatial_dimension() != 3:
            raise Exception, "Must be a tetrahedron"
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultTetrahedron( )
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A,self.b = reference_element.make_affine_mapping( v1 , v2 )       
        self.mapping = lambda x: numpy.dot( self.A , x ) + self.b
        self.scale = numpy.sqrt( numpy.linalg.det( self.A ) )

        return

    def get_num_members( self , n ):
        return (n+1)*(n+2)*(n+3)/6

    def tabulate( self , n , pts ):
        if len( pts ) == 0:
            return numpy.array( [] )


        ref_pts = [ self.mapping( pt ) for pt in pts ]

        def idx(p,q,r):
            return (p+q+r)*(p+q+r+1)*(p+q+r+2)/6 + (q+r)*(q+r+1)/2 + r

        def jrc( a , b , n ):
            an = float( ( 2*n+1+a+b)*(2*n+2+a+b)) \
                / float( 2*(n+1)*(n+1+a+b))
            bn = float( (a*a-b*b) * (2*n+1+a+b) ) \
                / float( 2*(n+1)*(2*n+a+b)*(n+1+a+b) )
            cn = float( (n+a)*(n+b)*(2*n+2+a+b)  ) \
                / float( (n+1)*(n+1+a+b)*(2*n+a+b) )
            return an,bn,cn

        apts = numpy.array( pts ) 

        results = numpy.zeros( ( (n+1)*(n+2)*(n+3)/6,len(pts)), type(pts[0][0]))
        results[0,:] = 1.0 + apts[:,0]-apts[:,0]+apts[:,1]-apts[:,1]+apts[:,2]-apts[:,2]

        if n == 0:
            return results

        x = numpy.array( [ pt[0] for pt in ref_pts ] )
        y = numpy.array( [ pt[1] for pt in ref_pts ] ) 
        z = numpy.array( [ pt[2] for pt in ref_pts ] )

        factor1 = 0.5 * ( 2.0 + 2.0*x + y + z ) 
        factor2 = (0.5*(y+z))**2
        factor3 = 0.5 * ( 1 + 2.0 * y + z )
        factor4 = 0.5 * ( 1 - z )
        factor5 = factor4 ** 2

        results[idx(1,0,0)] = factor1
        for p in range(1,n):
            a1 = ( 2.0 * p + 1.0 ) / ( p + 1.0 )
            a2 = p / (p + 1.0)
            results[idx(p+1,0,0)] = a1 * factor1 * results[idx(p,0,0)] \
                -a2 * factor2 * results[ idx(p-1,0,0) ]

        # q = 1
        for p in range(0,n):
            results[idx(p,1,0)] = results[idx(p,0,0)] \
                * ( p * (1.0 + y) + ( 2.0 + 3.0 * y + z ) / 2 )
            
        for p in range(0,n-1):
            for q in range(1,n-p):
                (aq,bq,cq) = jrc(2*p+1,0,q)
                qmcoeff = aq * factor3 + bq * factor4
                qm1coeff = cq * factor5
                results[idx(p,q+1,0)] = qmcoeff * results[idx(p,q,0)] \
                    - qm1coeff * results[idx(p,q-1,0)]

        # now handle r=1
        for p in range(n):
            for q in range(n-p):
                results[idx(p,q,1)] = results[idx(p,q,0)] \
                    * ( 1.0 + p + q + ( 2.0 + q + p ) * z )
 
        # general r by recurrence
        for p in range(n-1):
            for q in range(0,n-p-1):
                for r in range(1,n-p-q):
                    ar,br,cr = jrc(2*p+2*q+2,0,r)
                    results[idx(p,q,r+1)] = \
                                (ar * z + br) * results[idx(p,q,r) ] \
                                - cr * results[idx(p,q,r-1) ]

        for p in range(n+1):
            for q in range(n-p+1):
                for r in range(n-p-q+1):
                    results[idx(p,q,r)] *= math.sqrt((p+0.5)*(p+q+1.0)*(p+q+r+1.5))
        
        
        return results 

    def tabulate_jet( self , n , pts , order = 1 ):
        from Derivatives import DerivVar
        dpts = [ tuple( [ DerivVar( pt[i] , i , order ) \
                              for i in range( len( pt ) ) ] ) for pt in pts ]
        dbfs = self.tabulate( n , dpts )
        result = []
        for d in range( order + 1 ):
            result_d = [ [ foo[d] for foo in bar ] for bar in dbfs ]
            result.append( numpy.array( result_d ) )

        return result


def get_expansion_set( ref_el ):
    """Returns an ExpansionSet instance appopriate for the given
    reference element."""
    if ref_el.get_shape() == reference_element.LINE:
        return LineExpansionSet( ref_el )
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return TriangleExpansionSet( ref_el )
    elif ref_el.get_shape() == reference_element.TETRAHEDRON: 
        return TetrahedronExpansionSet( ref_el )
    else:
        raise Exception, "Unknown reference element type."

def polynomial_dimension( ref_el , degree ):
    """Returns the dimension of the space of polynomials of degree no
    greater than degree on the reference element."""
    if ref_el.get_shape() == reference_element.LINE:
        return max( 0 , degree + 1 )
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return max( (degree+1)*(degree+2)/2 , 0 )
    elif ref_el.get_shape() == reference_element.TETRAHEDRON: 
        return max( 0 , (degree+1)*(degree+2)*(degree+3)/6 )
    else:
        raise Exception, "Unknown reference element type."

if __name__=="__main__":
    import reference_element, expansions
    from FirstDerivatives import DerivVar

    E = reference_element.DefaultTriangle( )

    k = 3

    pts = E.make_lattice( k )

    dpts = [ [ DerivVar( pt[j] , j ) for j in range(len( pt )) ] for pt in pts ]

    Phis = expansions.get_expansion_set( E )

    phis = Phis.tabulate(k,pts)
    dphis = Phis.tabulate(k,dpts)


#    dphis_x = numpy.array( [ [ d[1][0] for d in dphi ] for dphi in dphis ] )
#    dphis_y = numpy.array([[d[1][1] for d in dphi ] for dphi in dphis ] )
#    dphis_z = numpy.array([[d[1][2] for d in dphi ] for dphi in dphis ] )

#    print dphis_x

    for dmat in make_dmats( E , k ):
        print dmat
        print

