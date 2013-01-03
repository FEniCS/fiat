# Copyright (C) 2005 The University of Chicago
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.
#
# Written by Robert C. Kirby
#
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650
#
# Last changed: 2005-05-16

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

    print(U.get_dual_set().entity_ids)
    print(U.get_nodal_basis().tabulate( T.make_lattice(1) ))



