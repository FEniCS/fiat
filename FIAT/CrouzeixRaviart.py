# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 1 June 2005


import shapes, dualbasis, polynomial, functional, functionalset

class CrouzeixRaviartDual( dualbasis.DualBasis ):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)"""
    def __init__( self , shape , U ):
        # in d dimensions, evaluate at midpoints of (d-1)-dimensional
        # entities
        d = shapes.dimension( shape )
        pts = [ pt for i in shapes.entity_range(shape,d-1) \
                for pt in shapes.make_points( shape , d-1 , i , d ) ]
        self.pts = pts
        ls = functional.make_point_evaluations( U , pts )
        entity_ids = {}
        for i in range(d-1):
            entity_ids[i] = {}
            for j in shapes.entity_range(shape,i):
                entity_ids[i][j] = []
        entity_ids[d-1] = {}
        for i in shapes.entity_range(shape,d-1):
            entity_ids[d-1][i] = [i]
        entity_ids[d] = { 0: [] }

        fset = functionalset.FunctionalSet( U , ls )
        
        dualbasis.DualBasis.__init__( self , fset , entity_ids )

class CrouzeixRaviart( polynomial.FiniteElement ):
    def __init__( self , shape, order = 1 ):
        self.shape = shape
        self.order = order
        if (order != 1) raise RuntimeError("Crouzeix-Raviart elements are only defined for order 1")
        U = polynomial.OrthogonalPolynomialSet( shape , 1 )
        Udual = CrouzeixRaviartDual( shape , U )
        polynomial.FiniteElement.__init__( self , Udual , U )
        return

class VectorCrouzeixRaviartDual( dualbasis.DualBasis ):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)"""
    def __init__( self , shape , U ):
        # in d dimensions, evaluate at midpoints of (d-1)-dimensional
        # entities
        nc = U.tensor_shape()[0]
        d = shapes.dimension(shape)
        pts = [ pt for i in shapes.entity_range(shape,d-1) \
                for pt in shapes.make_points( shape , d-1 , i , d ) ]
        self.pts = pts
        ls = [ functional.ComponentPointEvaluation( U , c, pt ) \
               for c in range(nc) \
               for pt in pts ]
        
        entity_ids = {}
        for i in range(d-1):
            entity_ids[i] = {}
            for j in shapes.entity_range(shape,i):
                entity_ids[i][j] = []
        entity_ids[d-1] = {}
        for i in shapes.entity_range(shape,d-1):
            entity_ids[d-1][i] = [i]
        entity_ids[d] = { 0: [] }

        fset = functionalset.FunctionalSet( U , ls )

        dualbasis.DualBasis.__init__( self , fset , entity_ids,nc )

class VectorCrouzeixRaviart( polynomial.FiniteElement ):
    def __init__( self , shape , order = 1, nc = None):
        self.shape = shape
        self.order = order
        if (order != 1) raise RuntimeError("Crouzeix-Raviart elements are only defined for order 1")
        U = polynomial.OrthogonalPolynomialArraySet( shape , 1, nc )
        Udual = VectorCrouzeixRaviartDual( shape , U )
        polynomial.FiniteElement.__init__( self , Udual , U )
        return

                
        
    
