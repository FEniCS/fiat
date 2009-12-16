# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last edited 16 May 2005
import reference_element, dual_set, numpy, functional, polynomial_set, finite_element

class P0Dual( dual_set.DualSet ):
    def __init__( self , ref_el ):
        entity_ids = {}
        nodes = []
        vs = numpy.array( ref_el.get_vertices() )
        bary=tuple( numpy.average( vs , 0 ) )
        
        nodes = [ functional.PointEvaluation( ref_el, bary ) ]
        entity_ids = { }
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        for dim in sorted( top ):
            entity_ids[dim] = {}
            for entity in sorted( top[dim] ):
                entity_ids[dim][entity] = []

        entity_ids[sd] = { 0 : [ 0 ] }
        
        dual_set.DualSet.__init__( self , nodes , ref_el , entity_ids )

class P0( finite_element.FiniteElement ):
    def __init__( self , ref_el ):
        poly_set = polynomial_set.ONPolynomialSet( ref_el , 0 )
        dual = P0Dual( ref_el )
        finite_element.FiniteElement.__init__( self , poly_set , dual , 0 )

if __name__ == "__main__":
    T = reference_element.UFCTriangle()
    U = P0( T )

    print U.get_dual_set().entity_ids
    print U.get_nodal_basis().tabulate( T.make_lattice(1) )



