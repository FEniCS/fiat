# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last modified 2 May 2005

import dualbasis, polynomial, functionalset, functional, shapes, \
       quadrature, Numeric, RaviartThomas

def NedelecSpace( k ):
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
    

# (P_k)^d \circplus { p \in (P_k^H)^d : p(x)\cdot x = 0 }
# Only defined on tetrahedra
# Arnold decomposes into curl^t of RT plus grad P_k
# indexint starts at zero

def NedelecSpaceRT( k ):
    shape = shapes.TETRAHEDRON
    d = shapes.dimension( shape )
    Vh = RaviartThomas.RTSpace( shape , k )

    Wh = polynomial.OrthogonalPolynomialSet( shape , k + 1 )

    curl_trans_rts = [ polynomial.curl_transpose( v ) \
                       for v in Vh ]
    
    curl_trans_rts_coeffs = Numeric.array( [ ctv.dof \
                                             for ctv in curl_trans_rts ] )
    curlTVh = polynomial.PolynomialSet( Vh.base , curl_trans_rts_coeffs )

    grad_whs = [ polynomial.gradient( w ) for w in Wh ]
    grad_whs_coeffs = Numeric.array( [ gw.dof for gw in grad_whs ] )
    gradWh = polynomial.PolynomialSet( Wh.base , grad_whs_coeffs )

    return polynomial.poly_set_union( curlTVh , gradWh )

class NedelecDual( dualbasis.DualBasis ):
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


        if k > 2:
            dim_Pk = shapes.polynomial_dimension( shape , k )
            vec_Pk = polynomial.OrthogonalPolynomialArraySet( shape , k )
            dim_Pkm3 = shapes.polynomial_dimension( shape , k-3 )
            vec_Pkm3 = vecPk.take( reduce( lambda a,b:a+b , \
                                           [ range( i*dim_Pk , \
                                                    i*dim_Pk+dim_Pkm3 ) \
                                             for i in range( d ) ] ) )
            interior_ls = [ functional.IntegralMoment( U , p ) \
                            for p in vec_Pkm3 ]
        else:
            interior_ls = []

        ls = edge_ls + face_ls + interior_ls

        cur = 0
        entity_ids = {}
        for i in range(d):
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

#        for e in entity_ids: print entity_ids[e]

        
        dualbasis.DualBasis.__init__( self ,
                                      functionalset.FunctionalSet( U , ls ) , \
                                      entity_ids )

 
class Nedelec( polynomial.FiniteElement ):
    def __init__( self , k ):
        U = NedelecSpace( k )
        Udual = NedelecDual( k , U )
        polynomial.FiniteElement.__init__( self , Udual , U )

