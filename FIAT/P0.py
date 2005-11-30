# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last edited 16 May 2005
import shapes, dualbasis, polynomial, Numeric, functional
import functionalset

class P0Dual( dualbasis.DualBasis ):
    def __init__( self , shape , U ):
        # get barycenter
        vs = shapes.vertices[shape]
        bary = Numeric.average( Numeric.array( map( Numeric.array , \
                                                   vs.values() ) ) )
        
        self.pts = ( tuple(bary) , )
        ls = [ functional.PointEvaluation( U , bary ) ]
        entity_ids = { }
        d = shapes.dimension( shape )
        for i in range(d):
            entity_ids[i] = {}
            for j in shapes.entity_range(shape,i):
                entity_ids[i][j] = {}
        entity_ids[d] = { 0 : [ 0 ] }

        fset = functionalset.FunctionalSet( U , ls )

        dualbasis.DualBasis.__init__( self , fset , entity_ids )

class P0( polynomial.FiniteElement ):
    def __init__( self , shape ):
        U = polynomial.OrthogonalPolynomialSet( shape , 0 )
        Udual = P0Dual( shape , U )
        polynomial.FiniteElement.__init__( self , Udual , U )

class VecP0Dual( dualbasis.DualBasis ):
    def __init__( self , shape , U ):
        # get barycenter
        d = shapes.dimension( shape )
        nc = U.tensor_shape()[0]
        vs = shapes.vertices[ shape ]
        bary = Numeric.average( Numeric.array( map( Numeric.array , \
                                                    vs.values() ) ) )
        self.pts = ( tuple(bary) , )
        ls = [ functional.ComponentPointEvaluation( U , c , bary ) \
               for c in range( nc ) ]
        entity_ids = {}
        for i in range(d):
            entity_ids[i] = {}
            for j in shapes.entity_range( shape , i ):
                entity_ids[i][j] = {}
        entity_ids[d] = { 0 : [ 0 ] }

        fset = functionalset.FunctionalSet( U , ls )

        dualbasis.DualBasis.__init__( self , fset , entity_ids , nc )

class VecP0( polynomial.FiniteElement ):
    def __init__( self , shape , nc ):
        U = polynomial.OrthogonalPolynomialArraySet( shape , 0 , nc )
        Udual = VecP0Dual( shape , U )
        polynomial.FiniteElement.__init__( self , Udual , U )

