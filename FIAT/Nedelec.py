# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last modified 2 May 2005

# Nedelec indexing from 0

import dualbasis, polynomial, functionalset, functional, shapes, \
       quadrature, Numeric, RaviartThomas

def NedelecSpace3D( k ):
    shape = shapes.TETRAHEDRON
    d = shapes.dimension( shape )
    vec_Pkp1 = polynomial.OrthogonalPolynomialArraySet( shape , k+1 )
    dimPkp1 = shapes.polynomial_dimension( shape , k+1 )
    dimPk = shapes.polynomial_dimension( shape , k )
    dimPkm1 = shapes.polynomial_dimension( shape , k-1 )
    vec_Pk = vec_Pkp1.take( reduce( lambda a,b:a+b , \
                                    [ range(i*dimPkp1,i*dimPkp1+dimPk) \
                                      for i in range(d) ] ) )
    vec_Pke = vec_Pkp1.take( reduce( lambda a,b:a+b , \
                                    [ range(i*dimPkp1+dimPkm1,i*dimPkp1+dimPk) \
                                      for i in range(d) ] ) )

    Pkp1 = polynomial.OrthogonalPolynomialSet( shape , k+1 )
    Q = quadrature.make_quadrature( shape , 2 * (k+1) )
    Pi = lambda f: polynomial.projection( Pkp1 , f , Q )
    PkCrossXcoeffs = Numeric.array( \
        [ [ Pi( lambda x: ( x[(i+2)%3] * p[(i+1)%3]( x ) \
                            - x[(i+1)%3] * p[(i+2)%3]( x ) ) ).dof \
            for i in range( d ) ] for p in vec_Pke ] )

    PkCrossX = polynomial.VectorPolynomialSet( Pkp1.base , PkCrossXcoeffs )
    return polynomial.poly_set_union( vec_Pk , PkCrossX )
   
def Nedelec02D():
	d = 2
	shape = shapes.TRIANGLE
	vec_P1 = polynomial.OrthogonalPolynomialArraySet(shape,1)
	dimP1 = 3
	dimP0 = 1
	vec_P0 = vec_P1.take(reduce(lambda a,b:a+b , \
						        [range(i*dimP1,i*dimP1+dimP0)\
						         for i in range(d)]))
	P1 = polynomial.OrthogonalPolynomialSet(shape,1)
	P0H=P1[:dimP0]
	Q=quadrature.make_quadrature(shape,2)
	P0Hrotxcoeffs = Numeric.array( \
	  [ [ polynomial.projection(P1,lambda x:x[1]*p(x),Q).dof , \
	      polynomial.projection(P1,lambda x:-x[0]*p(x),Q).dof ] \
	    for p in P0H ] )
	P0Hrotx = polynomial.VectorPolynomialSet(P1.base,P0Hrotxcoeffs)
	return polynomial.poly_set_union(vec_P0,P0Hrotx)
	    

def NedelecSpace2D( k ):
	if k==0:
		return Nedelec02D()
	shape = shapes.TRIANGLE
	d = shapes.dimension( shape )
	vec_Pkp1 = polynomial.OrthogonalPolynomialArraySet( shape , k+1 )
	dimPkp1 = shapes.polynomial_dimension( shape , k+1 )
	dimPk = shapes.polynomial_dimension( shape , k )
	dimPkm1 = shapes.polynomial_dimension( shape , k-1 )
	vec_Pk = vec_Pkp1.take( reduce( lambda a,b:a+b , \
                                    [ range(i*dimPkp1,i*dimPkp1+dimPk) \
                                      for i in range(d) ] ) )
	Pkp1     = polynomial.OrthogonalPolynomialSet( shape , k + 1 )
	PkH      = Pkp1[dimPkm1:dimPk]

	Q = quadrature.make_quadrature( shape , 2 * k )

	PkHrotxcoeffs = Numeric.array( \
    	[ [ polynomial.projection( Pkp1 , \
                                   lambda x:x[1]*p(x), Q ).dof , \
			polynomial.projection( Pkp1 , \
                                   lambda x:-x[0]*p(x), Q ).dof ] \
          for p in PkH ] )

	PkHrotx = polynomial.VectorPolynomialSet( Pkp1.base , PkHrotxcoeffs )

	return polynomial.poly_set_union( vec_Pk , PkHrotx )

def NedelecSpace( shape , degree ):
	if shape == shapes.TRIANGLE:
		return NedelecSpace2D( degree )
	elif shape == shapes.TETRAHEDRON:
		return NedelecSpace3D( degree )

class NedelecDual3D( dualbasis.DualBasis ):
    def __init__( self , U , k ):
		shape = shapes.TETRAHEDRON
		d = shapes.dimension( shape )
		ls = []
        # tangent at k+1 points on each edge
        
		edge_pts = [ shapes.make_points( shape , \
                                         1 , i , k+2 ) \
                     for i in shapes.entity_range( shape , \
                                                   1 ) ]
                                          
		mdcb = functional.make_directional_component_batch
		

		ls_per_edge = [ mdcb( U , \
                              shapes.tangents[shape][1][i] , \
                              edge_pts[i] ) \
                        for i in shapes.entity_range( shape , 1 ) ]


		edge_ls = reduce( lambda a,b:a+b , ls_per_edge )

        # tangential at dim(P_{k-1}) points per face	
		face_pts = [ shapes.make_points( shape , \
                                         2 , i , k+2 ) \
                     for i in shapes.entity_range( shape , \
                                                   2 ) ]

		ls_per_face = []
		for i in shapes.entity_range( shape , 2 ):
			ls_cur = []
			t0s = mdcb( U , shapes.tangents[shape][2][i][0] , face_pts[i] )
			t1s = mdcb( U , shapes.tangents[shape][2][i][1] , face_pts[i] )
			for i in range(len(t0s)):
				ls_cur.append( t0s[i] )
				ls_cur.append( t1s[i] )
			ls_per_face.append( ls_cur )
                        
		face_ls = reduce( lambda a,b:a+b , ls_per_face )


		if k > 1:
			dim_Pkp1 = shapes.polynomial_dimension( shape , k+1 )
			vec_Pkp1 = polynomial.OrthogonalPolynomialArraySet( shape , k+1 )
			dim_Pkm2 = shapes.polynomial_dimension( shape , k-2 )
			vec_Pkm2 = vec_Pkp1.take( reduce( lambda a,b:a+b , \
                                           [ range( i*dim_Pkp1 , \
                                                    i*dim_Pkp1+dim_Pkm2 ) \
                                             for i in range( d ) ] ) )
			interior_ls = [ functional.IntegralMoment( U , p ) \
                            for p in vec_Pkm2 ]
		else:
			interior_ls = []

		ls = edge_ls + face_ls + interior_ls

		cur = 0
		entity_ids = {}
		for i in range(d+1):
			entity_ids[i] = {}
			for j in shapes.entity_range(shape,i):
				entity_ids[i][j] = []

		nodes_per_edge = len( ls_per_edge[0] )
		nodes_per_face = len( ls_per_face[0] )
		internal_nodes = len( interior_ls )

        # loop over edges
		for i in shapes.entity_range(shape,1):
			for j in range( nodes_per_edge ):
				entity_ids[1][i].append( cur )
				cur += 1

		for i in shapes.entity_range(shape,2):
			for j in range(nodes_per_face):
				entity_ids[2][i].append( cur )
				cur += 1

		for j in range(len(interior_ls)):
			entity_ids[3][0].append( cur )
			cur += 1

		dualbasis.DualBasis.__init__( self ,
                                      ls , \
                                      entity_ids )

 
class NedelecDual2D( dualbasis.DualBasis ):
	def __init__( self , U , k ):
		shape = shapes.TRIANGLE
		mdcb = functional.make_directional_component_batch
		d = shapes.dimension( shape )
		pts_per_edge = [ [ x \
                           for x in shapes.make_points( shape , \
                                                        d-1 , \
                                                        i , \
                                                        d+k ) ] \
                        for i in shapes.entity_range( shape , d-1 ) ]
		tngnts = shapes.tangents[shapes.TRIANGLE][1]
		ls = reduce( lambda a,b:a+b , \
                     [ mdcb(U,tngnts[i],pts_per_edge[i]) \
                       for i in shapes.entity_range(shapes.TRIANGLE,1) ] )
		if k > 0:
			Pkp1 = polynomial.OrthogonalPolynomialArraySet( shape , k+1 )
			dim_Pkp1 = shapes.polynomial_dimension( shape , k+1 )
			dim_Pkm1 = shapes.polynomial_dimension( shape , k-1 )

			Pkm1 = Pkp1.take( reduce( lambda a,b:a+b , \
                                      [ range(i*dim_Pkp1,i*dim_Pkp1+dim_Pkm1) \
                                        for i in range(d) ] ) )
            

			interior_moments = [ functional.IntegralMoment( U , p ) \
                                 for p in Pkm1 ]
            
			ls.extend( interior_moments )
		else:
			interior_moments = []

		entity_ids = {}
		for i in range(d-1):
			entity_ids[i] = {}
			for j in shapes.entity_range(shape,i):
				entity_ids[i][j] = []
		pts_per_bdry = len(pts_per_edge[0])
		entity_ids[d-1] = {}
		node_cur = 0
		for j in shapes.entity_range(shape,d-1):
			for k in range(pts_per_bdry):
				entity_ids[d-1][j] = node_cur
				node_cur += 1
		entity_ids[d] = range(node_cur,\
                              node_cur+len(interior_moments))


		dualbasis.DualBasis.__init__( self , \
                                      functionalset.FunctionalSet( U , ls ) , \
                                      entity_ids )

def NedelecDual( shape , U , degree ):
	if shape == shapes.TRIANGLE:
		return NedelecDual2D( U , degree )
	elif shape == shapes.TETRAHEDRON:
		return NedelecDual3D( U , degree )


class Nedelec( polynomial.FiniteElement ):
    def __init__( self , shape , k ):
		print "Building Nedelec space"
		U = NedelecSpace( shape , k )
		print "Building dual space"
		Udual = NedelecDual( shape , U , k )
		
		polynomial.FiniteElement.__init__( self , Udual , U )

