# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 30 Nov 2005 (fix bug in P0 case for vectors)


"""This module defines the discontinuous Lagrange finite element over
any shape for which an appropriate expansion basis is defined.  The
module simply provides the class Lagrange, which takes a shape and a
degree >= 1.
"""

import dualbasis, functional, shapes, polynomial, functionalset, P0

class DiscLagrangeDual( dualbasis.DualBasis ):
    """Dual basis for the discontiuous Lagrange finite element."""
    def __init__( self , shape , n , U ):
        # Associate everything with the interior.
        self.pts = shapes.make_lattice( shape , n )

	ls = functional.make_point_evaluations( U , self.pts )

        # entity_ids is a dictionary whose keys are the topological
        # dimensions (0,1,2,3) and values are dictionaries mapping
        # the ids of entities of that dimension to the list of
        # ids of nodes associated with that entity.
        # see FIAT.base.dualbasis for description of entity_ids
        entity_ids = {}
        for d in range( shapes.dimension( shape ) ):
            entity_ids[d] = {}
        entity_ids[ shapes.dimension( shape ) ] = {}
        entity_ids[ shapes.dimension( shape ) ][ 0 ] = range( len( self.pts ) )
            
        dualbasis.DualBasis.__init__( self , \
                                      functionalset.FunctionalSet( U , ls ) , \
                                      entity_ids )
                                      
        return

class DiscVecLagrangeDual( dualbasis.DualBasis ):
    """Dual basis for the discontiuous Lagrange finite element."""
    def __init__( self , shape , n , U ):
        # Associate everything with the interior.
        d = shapes.dimension( shape )
        self.pts = shapes.make_lattice( shape , n )
        nc = U.tensor_shape()[0]

        ls = [ functional.ComponentPointEvaluation( U,c,pt ) \
               for c in range( nc ) for pt in self.pts ]

        # entity_ids is a dictionary whose keys are the topological
        # dimensions (0,1,2,3) and values are dictionaries mapping
        # the ids of entities of that dimension to the list of
        # ids of nodes associated with that entity.
        # see FIAT.base.dualbasis for description of entity_ids
        entity_ids = {}
        for dim in range( shapes.dimension( shape ) ):
            entity_ids[dim] = {}
            for e in shapes.entity_range( shape , dim ):
                entity_ids[dim][e] = {}
        entity_ids[ shapes.dimension( shape ) ] = {}
        entity_ids[ shapes.dimension( shape ) ][ 0 ] = range( len( self.pts ) )
            
        dualbasis.DualBasis.__init__( self , \
                                      functionalset.FunctionalSet( U , ls ) , \
                                      entity_ids , nc )
                                      
        return
    
class DiscLagrange( polynomial.FiniteElement ):
    def __init__( self , shape , n ):
        U = polynomial.OrthogonalPolynomialSet( shape , n )
        Udual = DiscLagrangeDual( shape , n , U )
        polynomial.FiniteElement.__init__( self , \
                                           Udual , U )

class DiscVecLagrange( polynomial.FiniteElement ):
    def __init__( self , shape , n , nc ):
        U = polynomial.OrthogonalPolynomialArraySet( shape , n , nc )
        Udual = DiscVecLagrangeDual( shape , n , U )
        polynomial.FiniteElement.__init__( self , \
                                           Udual , U )

def DiscontinuousLagrange( shape , n ):
    if n == 0: return P0.P0( shape )
    else:      return DiscLagrange( shape , n )

def DiscontinuousVectorLagrange( shape , n , nc ):
    if n == 0: return P0.VecP0( shape , nc )
    else:      return DiscVecLagrange( shape , n ,nc )
