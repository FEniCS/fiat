# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
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

import numpy

class DualSet:
    def __init__( self , nodes , ref_el , entity_ids ):
        self.nodes = nodes
        self.ref_el = ref_el
        self.entity_ids = entity_ids
        return

    def get_nodes( self ):
        return self.nodes

    def get_entity_ids( self ):
        return self.entity_ids

    def get_reference_element( self ):
        return self.ref_el

    def to_riesz( self , poly_set ):
        import time
        #print "riesz"

        # get an array of the right size, then write into it

        t1 = time.time()
        tshape = self.nodes[0].target_shape
        num_nodes = len( self.nodes )
        es = poly_set.get_expansion_set( )
        num_exp = es.get_num_members( poly_set.get_embedded_degree() )

        riesz_shape = tuple( [ num_nodes ] + list( tshape ) + [ num_exp ] )

        self.mat = numpy.zeros( riesz_shape , "d" )

        for i in range( len( self.nodes ) ):
            self.mat[i][:] = self.nodes[i].to_riesz( poly_set )

        #print "time new: ", time.time() - t1

        t1 = time.time()
#        from functional import Functional
#        riesz_reps = [ Functional.to_riesz( n , poly_set ) for n in self.nodes ]
#        print "time old: ", time.time() - t1


        #print "done with riesz"

        return self.mat
