# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 9 may 2005


import shapes, points, dualbasis, polynomial, functional, functionalset

class CrouzeixRaviartDual( dualbasis.DualBasis ):
    """Dual basis for Crouzeix-Raviart element (linears continuous at
    boundary midpoints)"""
    def __init__( self , shape , U ):
        # in d dimensions, evaluate at midpoints of (d-1)-dimensional
        # entities
        d = shapes.dims[ shape ]
        pts = [ pt for i in shapes.entity_range(shape,d-1) \
                for pt in points.make_points( shape , d-1 , i , d ) ]
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
    def __init__( self , shape ):
        U = polynomial.OrthogonalPolynomialSet( shape , 1 )
        Udual = CrouzeixRaviartDual( shape , U )
        polynomial.FiniteElement.__init__( self , Udual , U )
        return

        
                
        
    
