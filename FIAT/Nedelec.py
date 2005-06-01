# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last modified 2 May 2005

import dualbasis, polynomial, functionalset, functional, shapes_new, \
       quadrature, Numeric, RaviartThomas

# (P_k)^d \circplus { p \in (P_k^H)^d : p(x)\cdot x = 0 }
# Only defined on tetrahedra
# Arnold decomposes as curl^t of RT plus grad P_k
# indexint starts at zero

def NedelecSpace( k ):
    shape = shapes_new.TETRAHEDRON
    d = shapes_new.dimension( shape )
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
        shape = shapes_new.TETRAHEDRON
        # tangent at k+1 points on each edge
        
        edge_pts = [ shapes_new.make_points( shape , \
                                         1 , i , k+2 ) \
                     for i in shapes_new.entity_range( shape , \
                                                   1 ) ]

        mdcb = functional.MakeDirectionalComponentBatch
        ls_per_edge = [ mdcb( edge_pts[i] , \
                              shapes_new.tangents[shape][i] ) \
                        for i in shape.entity_range( shape , 1 ) ]

        edge_ls = reduce( lambda a,b:a+b , ls_per_edge )

        # cross with normal at dim(P_{k-1}) points per face
        face_pts = [ shapes_new.make_points( shape , \
                                         2 , i , k+1 ) \
                     for i in shapes_new.entity_range( shape , \
                                                   2 ) ]

        # internal moments of dim( P_{k-2}) points
        internal_pts = shapes_new.make_points( shape , \
                                           3 , i , k+2 ) 

        entity_ids = {}
        entity_ids[0] = {}
        entity_ids[1] = {}
        cur = 0
        for i in shapes_new.entity_range(shape,1):
            entity_ids[1][i] = []
            for j in range(k+1):
                entity_ids[1][i].append(cur)
                cur += 1
        entity_ids[2] = {}
        for i in shapes_new.entity_range(shape,2):
            entity_ids[2][i] = []
            for j in range(len(face_pts[0])):
                entity_ids[2][i].append[cur]
                cur += 1

        entity_ids[3] = {}
        entity_ids[3][0] = []
        for i in len( face_pts ):
            entity_ids[3][0].append( cur )
            cur += 1

 
class Nedelec( polynomial.FiniteElement ):
    def __init__( self , k ):
        U = NedelecSpace( k )
        Udual = NedelecDual( k , U )
        polynomial.FiniteElement.__init__( self , Udual , U )

