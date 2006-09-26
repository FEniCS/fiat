# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last edited 1 June 2005

import dualbasis, polynomial, functionalset, functional, PhiK, shapes

# do bulk ON eval on edges
class BDMDualBulk1( dualbasis.DualBasis ):
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

        interior_moments = []
        
        # internal moments against gradients of polynomials
        # of degree k-1 (only if k > 1)

        if k > 1:
            pk = polynomial.OrthogonalPolynomialSet(shape,k)
            pkm1 = pk[1:shapes.polynomial_dimension(shape,k-1)]
    
            pkm1grads = [ polynomial.gradient( p ) for p in pkm1 ]

            interior_moments.extend( [ functional.IntegralMoment( U , pg ) \
                                       for pg in pkm1grads ] )

        # internal moments against div-free polynomials with
        # vanishing normal component (only if n > 2)
        if k > 1:
            PHIK = PhiK.PhiK( shape , k , U )

            interior_moments.extend( [ functional.IntegralMoment( U ,  phi ) \
                                       for phi in PHIK ] )

        ls.extend( interior_moments )
        
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

class BDMDualBulk2( dualbasis.DualBasis ):
    def __init__( self , shape , k , U ):
        mdcb = functional.make_directional_component_batch
        d = shapes.dimension( shape )
        pts_per_edge = [ [ x \
                           for x in make_points( shape , \
                                                        d-1 , \
                                                        i , \
                                                        d+k ) ] \
                        for i in shapes.entity_range( shape , d-1 ) ]
        nrmls = shapes.normals[shape]
        ls = reduce( lambda a,b:a+b , \
                     [ mdcb(U,nrmls[i],pts_per_edge[i]) \
                       for i in shapes.entity_range(shape,d-1) ] )

        interior_moments = []
        
        # internal moments against gradients of polynomials
        # of degree k-1 (only if k > 1)

        if k > 1:
            pk = polynomial.OrthogonalPolynomialSet(shape,k)
            pkm1grads = polynomial.gradients( \
                pk[1:shapes.polynomial_dimension(shape,k-1)] )

            interior_moments.extend( [ functional.Functional(U,dofs) \
                                       for dofs in pkm1grads.coeffs ] )

        # internal moments against div-free polynomials with
        # vanishing normal component (only if n > 2)
        if k > 1:
            PHIK = PhiK.PhiK( shape , k , U )

            interior_moments.extend( [ functional.IntegralMoment( U ,  phi ) \
                                       for phi in PHIK ] )

        ls.extend( interior_moments )
        
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


class BDMDual( dualbasis.DualBasis ):
    def __init__( self , shape , k , U ):
        # normal components on the edges/faces
        DCPE = functional.DirectionalComponentPointEvaluation
        d = shapes.dimension( shape )
        pts_per_edge = [ [ x \
                           for x in shapes.make_points( shape , \
                                                        d-1 , \
                                                        i , \
                                                        d+k ) ] \
                        for i in shapes.entity_range( shape , d-1 ) ]
        pts_flat = reduce( lambda a,b:a+b , pts_per_edge )
        ls = [ ]
        for i in range(d+1):
            nrml = shapes.normals[ shape ][ i ]
            ls_cur = [ DCPE(U,nrml,pt) for pt in pts_per_edge[i] ]
            ls.extend(ls_cur)

        interior_moments = []
        
        # internal moments against gradients of polynomials
        # of degree k-1 (only if k > 1)

        if k > 1:
            pk = polynomial.OrthogonalPolynomialSet(shape,k)
            pkm1 = pk[1:shapes.polynomial_dimension(shape,k-1)]
    
            pkm1grads = [ polynomial.gradient( p ) for p in pkm1 ]

            interior_moments.extend( [ functional.IntegralMoment( U , pg ) \
                                       for pg in pkm1grads ] )

        # internal moments against div-free polynomials with
        # vanishing normal component (only if n > 2)
        if k > 1:
            PHIK = PhiK.PhiK( shape , k , U )

            interior_moments.extend( [ functional.IntegralMoment( U ,  phi ) \
                                       for phi in PHIK ] )

        ls.extend( interior_moments )
        
        entity_ids = {}
        for i in range(d-1):
            entity_ids[i] = {}
            for j in shapes.entity_range(shape,i):
                entity_ids[i][j] = []
        pts_per_bdry = len(pts_per_edge[0])
        entity_ids[d-1] = {}
        node_cur = 0
        for j in shapes.entity_range(shape,d-1):
            entity_ids[d-1][j] = range(node_cur,node_cur+pts_per_bdry)
            node_cur += pts_per_bdry
        entity_ids[d] = range(node_cur,\
                              node_cur+len(interior_moments))

        dualbasis.DualBasis.__init__( self , \
                                      functionalset.FunctionalSet( U , ls ) , \
                                      entity_ids )


class BDM( polynomial.FiniteElement ):
    def __init__( self , shape , n ):
        U = polynomial.OrthogonalPolynomialArraySet( shape , n )
        Udual = BDMDual( shape , n , U )
        polynomial.FiniteElement.__init__( self , Udual , U )

class BDMBulk1( polynomial.FiniteElement ):
    def __init__( self , shape , n ):
        U = polynomial.OrthogonalPolynomialArraySet( shape , n )
        Udual = BDMDualBulk1( shape , n , U )
        polynomial.FiniteElement.__init__( self , Udual , U )

class BDMBulk2( polynomial.FiniteElement ):
    def __init__( self , shape , n ):
        U = polynomial.OrthogonalPolynomialArraySet( shape , n )
        Udual = BDMDualBulk2( shape , n , U )
        polynomial.FiniteElement.__init__( self , Udual , U )
