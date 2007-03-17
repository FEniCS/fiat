# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 1 June 2005


"""This module defines the standard Lagrange finite element over
any shape for which an appropriate expansion basis is defined.  The
module simply provides the class Lagrange, which takes a shape and a
degree >= 1.
"""

import dualbasis, functional, polynomial, functionalset , shapes

class LagrangeDual( dualbasis.DualBasis ):
    """Dual basis for the plain vanilla Lagrange finite element."""
    def __init__( self , shape , n , U ):
        # Nodes are point evaluation on the lattice, which here is
        # ordered by topological dimension and entity within that
        # dimension.  For triangles we have vertex points plus
        # n - 1 points per edge.  We then have max(0,(n-2)(n-1)/2)
        # points on the interior of the triangle, or that many points
        # per face of tetrahedra plus max(0,(n-2)(n-1)n/6 points
        # in the interior
        pts = [ [ shapes.make_points(shape,d,e,n) \
                      for e in shapes.entity_range(shape,d) ] \
                    for d in shapes.dimension_range(shape) ]

        pts_flat = reduce( lambda a,b:a+b , \
                           reduce( lambda a,b:a+b , pts ) )

        # get node location for each node i by self.pts[i]
        self.pts = pts_flat

	ls = functional.make_point_evaluations( U , pts_flat )

        # entity_ids is a dictionary whose keys are the topological
        # dimensions (0,1,2,3) and values are dictionaries mapping
        # the ids of entities of that dimension to the list of
        # ids of nodes associated with that entity.
        # see FIAT.base.dualbasis for description of entity_ids
        entity_ids = {}
        id_cur = 0
        for d in shapes.dimension_range( shape ):
            entity_ids[d] = {}
            for v in shapes.entity_range( shape, d ):
                num_nods = len( pts[d][v] )
                entity_ids[d][v] = range(id_cur,id_cur + num_nods)
                id_cur += num_nods

        dualbasis.DualBasis.__init__( self , \
                                      functionalset.FunctionalSet( U , ls ) , \
                                      entity_ids )
                                      
        return

class Lagrange( polynomial.FiniteElement ):
    def __init__( self , shape , n ):
        if n < 1:
            raise RuntimeError, \
                  "Lagrange elements are only defind for n >= 1"
        self.shape = shape
        self.order = n
        U = polynomial.OrthogonalPolynomialSet( shape , n )
        Udual = LagrangeDual( shape , n , U )
        polynomial.FiniteElement.__init__( self , \
                                           Udual , U )

class VectorLagrangeDual( dualbasis.DualBasis ):
    def __init__( self , shape , n , U ):
	space_dim = shapes.dimension( shape )
        nc = U.tensor_shape()[0]
        pts = [ [ shapes.make_points(shape,d,e,n) \
                  for e in shapes.entity_range(shape,d) ] \
                for d in shapes.dimension_range(shape) ]

        pts_flat = reduce( lambda a,b:a+b , \
                           reduce( lambda a,b:a+b , pts ) )

        self.pts = pts_flat
	
        ls = [ functional.ComponentPointEvaluation( U , c , pt ) \
	       for c in range(nc) \
	       for pt in pts_flat ]
        

        # see FIAT.base.dualbasis for description of entity_ids
        entity_ids = {}
        id_cur = 0
        for d in shapes.dimension_range( shape ):
            entity_ids[d] = {}
            for v in shapes.entity_range( shape, d ):
                num_nods = len( pts[d][v] )
                entity_ids[d][v] = range(id_cur,id_cur + num_nods)
                id_cur += num_nods
        
        fset = functionalset.FunctionalSet( U , ls )

        dualbasis.DualBasis.__init__( self , fset , entity_ids , nc ) 


class VectorLagrange( polynomial.FiniteElement ):
    def __init__( self , shape , n , nc = None ):
        if n < 1:
            raise RuntimeError, \
                  "Lagrange elements are only defined for n >= 1"
        self.shape = shape
        self.order = n
        U = polynomial.OrthogonalPolynomialArraySet( shape , n , nc )
        Udual = VectorLagrangeDual( shape , n , U )
        polynomial.FiniteElement.__init__( self , Udual , U )
