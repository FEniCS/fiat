# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650
#
# Last modified 10 May 2005 by RCK

import functional, numpy

#class DualBasis( functional.FunctionalList ):
class DualBasis( object ):
    '''Creates the basis P\' which is dual to some space P.
    ls is a list of functionals or a FunctionalList.
    entity_ids is a dictionary mapping each mesh component to
    the associated vertex numbers.  The keys are integers for
    the topological dimension, and each value is itself a
    dictionary mapping the id of each entity of that topological
    dimension to a list of ids for functionals associated with that
    entity. E.g. for linear Lagrange elements on triangles, entity_ids
    should look like:
    { 0: {0: [0], 1: [1], 2: [2]}, \
      1: {0: [] , 2: [], 2: []}, \
      2: {0: []} }
    since there are three vertices (dimension 0),
    each with one node each, and there are no nodes on the edges or
    interior.  Similarly, cubics would look like:
    { 0: {0: [0], 1:[1], 2:[2]}, \
      1: {0: [3,4], 1:[5,6], 2:[7,8]}, \
      2: {0: [9]} }
    num_reps is used to indicate a vector-valued space where
    entity_ids is used for the first component and then
    replicated num_reps-1 times additional times.
    '''
    def __init__( self, node_set , entity_ids , num_reps = 1 ):
        self.node_set = node_set
        self.entity_ids = entity_ids
        self.num_reps = num_reps
	# self.mat is the matrix whose rows are the vectors
	# associated with each functional
	# Now, all the dual basis can be applied to a single
	# item by a matrix-vector multiply
        return
    def get_functional_set( self ): return self.node_set
    def getNodeIDs(self, dim):
        try:
            return self.entity_ids[ dim ].values()
        except:
            raise RuntimeError, "Illegal dimension for this dual basis"

    def get_dualbasis_types(self):
        types = [self.node_set[i].get_type()
                 for i in range(len(self.node_set))]
        return types
