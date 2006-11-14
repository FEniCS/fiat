# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last modified 2 May 2005

import dualbasis, polynomial, functionalset, functional, shapes, \
       quadrature, numpy


def RT0Space( shape ):
    d = shapes.dimension( shape )
    vec_P1 = polynomial.OrthogonalPolynomialArraySet( shape , 1 )
    dimP1 = shapes.polynomial_dimension( shape , 1 )
    dimP0 = shapes.polynomial_dimension( shape , 0 )

    vec_P0 = vec_P1.take( reduce( lambda a,b:a+b , \
                                  [ range(i*dimP1,i*dimP1+dimP0) \
                                    for i in range(d) ] ) )
    P1 = polynomial.OrthogonalPolynomialSet( shape , 1 )
    P0H = P1[:dimP0]

    Q = quadrature.make_quadrature( shape , 2 )

    P0Hxcoeffs = numpy.array( [ [ polynomial.projection( P1 , \
                                                           lambda x:x[i]*p(x), \
                                                           Q ).dof \
                                    for i in range(d) ] \
                                    for p in P0H ] )

    P0Hx = polynomial.VectorPolynomialSet( P1.base , P0Hxcoeffs )

    return polynomial.poly_set_union( vec_P0 , P0Hx )
    

# (P_k)^2 + x (P_k)
def RTSpace( shape , k ):
    if k == 0:
        return RT0Space( shape )
    d = shapes.dimension( shape )
    vec_Pkp1 = polynomial.OrthogonalPolynomialArraySet( shape , k+1 )
    dimPkp1  = shapes.polynomial_dimension( shape , k+1 )
    dimPk    = shapes.polynomial_dimension( shape , k )
    dimPkm1  = shapes.polynomial_dimension( shape , k-1 )
    vec_Pk   = vec_Pkp1.take( reduce( lambda a,b:a+b , \
                                      [ range(i*dimPkp1,i*dimPkp1+dimPk) \
                                        for i in range(d) ] ) )
    Pkp1     = polynomial.OrthogonalPolynomialSet( shape , k + 1 )
    PkH      = Pkp1[dimPkm1:dimPk]

    Q = quadrature.make_quadrature( shape , 2 * k )

    PkHxcoeffs = numpy.array( [ [ polynomial.projection( Pkp1 , \
                                                           lambda x:x[i]*p(x), \
                                                           Q ).dof \
                                    for i in range(d) ] \
                                    for p in PkH ] )

    PkHx = polynomial.VectorPolynomialSet( Pkp1.base , PkHxcoeffs )

    return polynomial.poly_set_union( vec_Pk , PkHx )


class RTDual( dualbasis.DualBasis ):
    def __init__( self , shape , k , U ):
        mdcb = functional.make_directional_component_batch
        d = shapes.dimension( shape )
        pts_per_edge = [ [ x \
                           for x in shapes.make_points( shape , \
                                                        d-1 , \
                                                        i , \
                                                        d+k ) ] \
                        for i in shapes.entity_range( shape , d-1 ) ]
        nrmls = shapes.normals[shape]
        ls = reduce( lambda a,b:a+b , \
                     [ mdcb(U,nrmls[i],pts_per_edge[i]) \
                       for i in shapes.entity_range(shape,d-1) ] )

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

class RT0( polynomial.FiniteElement ):
    def __init__( self , shape ):
        U = RT0Space( shape )
        Udual = RTDual( shape , 0 , U )
        polynomial.FiniteElement.__init__( self , Udual , U )


class RaviartThomas( polynomial.FiniteElement ):
    def __init__( self , shape , n ):
        U = RTSpace( shape , n )
        Udual = RTDual( shape , n , U )
        polynomial.FiniteElement.__init__( self , Udual , U )

