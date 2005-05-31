# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last modified 7 April 2005

import Numeric
from LinearAlgebra import singular_value_decomposition as svd
from polynomial import outer_product

class FunctionalSet( object ):
    """A functional set is a list of linear functionals
    where we store all the coefficients in a matrix together."""
    def __init__( self , U , ls ):
        # confirm that all functionals are over the same space
        isU0 = [ l.U is ls[0].U for l in ls[1:] ]
        # need default value for picky case of single node
        if not reduce( lambda a,b : a and b , isU0 , True ):
            raise RuntimeError, "Not all functionals have the same base!"
        self.U = U
        self.ls = ls
	self.mat = Numeric.array( [ Numeric.array( l.a ) for l in ls ] )
        pass
    def __getitem__( self , i ):
        """Recovers the i:th functional in the set."""
        return self.ls[i]
    def get_matrix( self ):
        return self.mat
    def function_space( self ): return self.U
    def __len__( self ): return len( self.ls )
