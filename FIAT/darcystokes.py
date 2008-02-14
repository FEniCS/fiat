# implement a general family of elements extending the Darcy-Stokes element
from FIAT.polynomial import ConstrainedPolynomialSet, OrthogonalPolynomialSet, \
	OrthogonalPolynomialArraySet, outer_product
from FIAT import shapes
from FIAT.functional import FacetDirectionalMoment as fdm
from FIAT.functional import IntegralMomentOfDivergence, IntegralMoment
from FIAT.functionalset import FunctionalSet
from FIAT.dualbasis import DualBasis
from FIAT.functional import make_directional_component_batch as mdcb
from numpy.linalg import svd
from FIAT import polynomial

# Space of vectors of polynomials of degree k+3.
# Normal components are restricted to degree k+1
# The divergence is restricted to degree k
def DarcyStokesSpace( k ):
    if k%2:
        raise Exception, "Only works for even order"

    shape = shapes.TRIANGLE
    d = shapes.dimension( shape )
    U = OrthogonalPolynomialArraySet( shape , k + 3 )

    constraints = []
        
    # set up edge constraints
    phi_edge = OrthogonalPolynomialSet( shape - 1 , k + 3 )
    dimPkp1_edge = shapes.polynomial_dimension( shape - 1 , k + 1)

    for e in shapes.entity_range( shape , d - 1 ):
	n = shapes.normals[ shape ][ e ]
	edge_constraints = [ fdm( U , shape , n , d-1 , e , phi ) \
	    			 for phi in phi_edge[dimPkp1_edge:] ]
	constraints.extend( edge_constraints )
	
    # get constraints on divergence in the interior
    phi_interior = OrthogonalPolynomialSet( shape , k + 3 )
    phi_start = shapes.polynomial_dimension( shape , k )
    phi_finish = shapes.polynomial_dimension( shape , k + 2 )
    divergence_constraints = [ IntegralMomentOfDivergence( U , phi ) \
    			       for phi in phi_interior[phi_start:phi_finish] ]
			
    constraints.extend( divergence_constraints )		
	
    fset = FunctionalSet( U , constraints )
    return ConstrainedPolynomialSet( fset )


def DarcyStokesSpaceOdd( k ):
    if not k%2:
        raise Exception, "Only works for odd order"

    shape = shapes.TRIANGLE
    d = shapes.dimension( shape )
    U = OrthogonalPolynomialArraySet( shape , k + 3 )

    constraints = []
        
    # set up edge constraints
    phi_edge = OrthogonalPolynomialSet( shape - 1 , k + 3 )
    dimPkp2_edge = shapes.polynomial_dimension( shape - 1 , k +  2)

    for e in shapes.entity_range( shape , d - 1 ):
	n = shapes.normals[ shape ][ e ]
	edge_constraints = [ fdm( U , shape , n , d-1 , e , phi ) \
	    			 for phi in phi_edge[dimPkp2_edge:] ]
	constraints.extend( edge_constraints )
	
    # get constraints on divergence in the interior
    phi_interior = OrthogonalPolynomialSet( shape , k + 3 )
    phi_start = shapes.polynomial_dimension( shape , k )
    phi_finish = shapes.polynomial_dimension( shape , k + 2 )
    divergence_constraints = [ IntegralMomentOfDivergence( U , phi ) \
    			       for phi in phi_interior[phi_start:phi_finish] ]
			
    constraints.extend( divergence_constraints )		
	
    fset = FunctionalSet( U , constraints )
    return ConstrainedPolynomialSet( fset )
    

class DarcyStokesDual( DualBasis ):
    def __init__( self , k , U ):
	shape = shapes.TRIANGLE
	d = shapes.dimension( shapes.TRIANGLE )

	normals = shapes.normals[ shapes.TRIANGLE ] 
	tangents = shapes.tangents[ shapes.TRIANGLE ][ shapes.LINE ]

	normal_pts_per_edge = [ shapes.make_points( shape , d-1 , i , d+k+1 ) \
				for i in shapes.entity_range( shape , d - 1 ) ]

	normal_ls = reduce( lambda a,b: a + b , \
			[ mdcb( U , normals[i] , normal_pts_per_edge[i] ) \
			  for i in shapes.entity_range( shape , d - 1 ) ] )

	tangential_pts_per_edge = [ shapes.make_points( shape , d-1 , i , d+k ) \
				for i in shapes.entity_range( shape , d - 1 ) ]

	tangential_ls = reduce( lambda a,b: a + b , \
				[ mdcb( U , tangents[i] , tangential_pts_per_edge[i] ) \
				  for i in shapes.entity_range( shape , d - 1 ) ] )
	
	#internal dof are like RT3
	# base of space is P_{k+3}, so I start with this
	# and extract the pieces that are degree k-1
	if k > 0:
            Pkp3 = OrthogonalPolynomialArraySet( shape , k + 3 )
            dimPkp3 = shapes.polynomial_dimension( shape , k + 3 )
	    dimPkm1 = shapes.polynomial_dimension( shape , k - 1 )
	    Pkm1 = Pkp3.take( reduce( lambda a,b:a+b , \
	       		      [ range( i * dimPkp3 , i*dimPkp3+dimPkm1) \
				for i in range( d ) ] ) )
	    interior_ls = [ IntegralMoment( U , p ) for p in Pkm1 ]
	else: 
            interior_ls = []

	ls = normal_ls + tangential_ls + interior_ls

        entity_ids = {}

        # no vertex dof
        entity_ids[0] = {}
        for j in shapes.entity_range( shape , 0 ):
            entity_ids[0][j] = []

        # edge dof: have all normal dof then all tangential
        # have k+2 dof normals per edge, then k+1 tangentials
        # total of 3*(k+2) normals followed by 3*(k+1) tangents
        entity_ids[1] = {}

        for j in shapes.entity_range( shape , 1 ):
            entity_ids[1][j] = range((k+2)*j,(k+2)*(j+1))\
                               +range(3*(k+2)+(k+1)*j,(3*(k+2)+(k+1)*(j+1)))

        entity_ids[2] = {}
        entity_ids[2][0] = range( 3*(2*k+3) , len( ls ) )

	DualBasis.__init__( self , FunctionalSet( U , ls ) , entity_ids )

class DarcyStokes( polynomial.FiniteElement ):
    def __init__( self , n ):
        U = DarcyStokesSpace( n )
        Udual = DarcyStokesDual( n , U )
        polynomial.FiniteElement.__init__( self , Udual , U )
        return

class DarcyStokesDualOdd( DualBasis ):
    def __init__( self , k , U ):
	shape = shapes.TRIANGLE
	d = shapes.dimension( shapes.TRIANGLE )

	normals = shapes.normals[ shapes.TRIANGLE ] 
	tangents = shapes.tangents[ shapes.TRIANGLE ][ shapes.LINE ]

	normal_pts_per_edge = [ shapes.make_points( shape , d-1 , i , d+k+1 ) \
				for i in shapes.entity_range( shape , d - 1 ) ]

	normal_ls = reduce( lambda a,b: a + b , \
			[ mdcb( U , normals[i] , normal_pts_per_edge[i] ) \
			  for i in shapes.entity_range( shape , d - 1 ) ] )

	tangential_pts_per_edge = [ shapes.make_points( shape , d-1 , i , d+k+1 ) \
				for i in shapes.entity_range( shape , d - 1 ) ]

	tangential_ls = reduce( lambda a,b: a + b , \
				[ mdcb( U , tangents[i] , tangential_pts_per_edge[i] ) \
				  for i in shapes.entity_range( shape , d - 1 ) ] )
	
	#internal dof are like RT3
	# base of space is P_{k+3}, so I start with this
	# and extract the pieces that are degree k-1
	if k > 0:
            Pkp3 = OrthogonalPolynomialArraySet( shape , k + 3 )
            dimPkp3 = shapes.polynomial_dimension( shape , k + 3 )
	    dimPkm1 = shapes.polynomial_dimension( shape , k - 1 )
	    Pkm1 = Pkp3.take( reduce( lambda a,b:a+b , \
	       		      [ range( i * dimPkp3 , i*dimPkp3+dimPkm1) \
				for i in range( d ) ] ) )
	    interior_ls = [ IntegralMoment( U , p ) for p in Pkm1 ]
	else: 
            interior_ls = []

	ls = normal_ls + tangential_ls + interior_ls

        entity_ids = {}

        # no vertex dof
        entity_ids[0] = {}
        for j in shapes.entity_range( shape , 0 ):
            entity_ids[0][j] = []

        # edge dof: have all normal dof then all tangential
        # have k+2 dof normals per edge, then k+1 tangentials
        # total of 3*(k+2) normals followed by 3*(k+2) tangents
        entity_ids[1] = {}

        for j in shapes.entity_range( shape , 1 ):
            entity_ids[1][j] = range((k+2)*j,(k+2)*(j+1))\
                               +range(3*(k+2)+(k+2)*j,(3*(k+2)+(k+2)*(j+1)))

        entity_ids[2] = {}
        entity_ids[2][0] = range( 3*(2*k+3) , len( ls ) )

	DualBasis.__init__( self , FunctionalSet( U , ls ) , entity_ids )


if __name__ == "__main__":
    print "testing even"
    for k in range( 0 , 4 , 2 ):
        U = DarcyStokesSpace(k)
	print "dim of space of order " , k , " " , len( U )
	Udual = DarcyStokesDual( k , U )
	V = outer_product( Udual.get_functional_set() , U )
	(u,s,vt) = svd( V )
	print s
#	print Udual.entity_ids
    print "testing odd"
    for k in range( 1 , 5 , 2 ):
        U = DarcyStokesSpaceOdd( k )
        print "dim of space of order " , k , " " , len( U )
        Udual = DarcyStokesDualOdd( k , U )
        print "dim of dual " , len( Udual.get_functional_set() )
        V = outer_product( Udual.get_functional_set() , U )
        (u,s,vt) = svd( V )
        print s

		
