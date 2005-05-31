# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 7 Feb 2005 by RCK

"""shapes.py
Module defining topological information for simplices in one, two, and
three dimensions."""
from math import sqrt
import exceptions
from factorial import factorial

rt2 = sqrt(2.0)
rt3 = sqrt(3.0)
rt2_inv = 1./rt2
rt3_inv = 1./rt3


class ShapeError( exceptions.Exception ):
    """FIAT defined exception type indicating an improper use of a shape."""
    def __init__( self , args = None ):
        self.args = args
        return

# enumerated constants for shapes
LINE = 1
TRIANGLE = 2
TETRAHEDRON = 3

# the dimension associated with each shape
dims = { LINE: 1 , \
         TRIANGLE: 2 , \
         TETRAHEDRON: 3 }

# number of topological entities of each dimensions
# associated with each shape.
# 0 refers to the number of points that make up the simplex
# 1 refers to the number of edges
# 2 refers to the number of triangles/faces
# 3 refers to the number of tetrahedra
num_entities = { LINE: { 0: 2 , 1: 1 } , \
                 TRIANGLE: { 0: 3, 1: 3, 2:1 } , \
                 TETRAHEDRON: {0: 4, 1: 6, 2: 4, 3: 1} }

# canonical polynomial dimension associated with simplex
# of each dimension
poly_dims = { LINE: lambda n: max(0,n + 1), \
              TRIANGLE: lambda n: max(0,(n+1)*(n+2)/2) ,\
              TETRAHEDRON: lambda n: max(0,(n+1)*(n+2)*(n+3)/6) }


# dictionaries of vertices
vertices = { LINE : {0: (-1.0,) , 1: (1.0,)} , \
             TRIANGLE : {0:(-1.0,-1.0),\
                         1:(1.0,-1.0),\
                         2:(-1.0,1.0)} , \
             TETRAHEDRON :{0:(-1.,-1.,-1.), \
                           1:(1.,-1.,-1.), \
                           2:(-1.,1.,-1.), \
                           3:(-1.,-1.,1.)} }


# FYI -- this is the ordering scheme I'm thinking of.
# Should be verified against tetgen or
# whatever mesh generator we wind up using
tetrahedron_faces = {0:(1,2,3), \
                     1:(0,3,2), \
                     2:(0,1,3), \
                     3:(0,2,1)}

tetrahedron_edges = {0:(1,2), \
                     1:(2,3), \
                     2:(3,1), \
                     3:(3,0), \
                     4:(0,2), \
                     5:(0,1)}

# scaling factors mapping boundary entities of lower dimension to the reference
# shape of that dimension.  For example, in 2d, two of the edges have length
# 2, which is the same length as the reference line.  The third is 2 sqrt(2),
# so we get two scale factors of 1 and one of sqrt(2.0)
jac_factors = { LINE: { 0: { 0: 1.0, \
                             1: 1.0 } , \
                        1: { 0: 1.0 } } , \
                TRIANGLE: { 0: { 0: 1.0 , \
                                 1: 1.0 , \
                                 2: 1.0 } , \
                            1: { 0: rt2 , \
                                 1: 1.0 , \
                                 2: 1.0 } , \
                            2: { 0: 1.0 } } , \
                TETRAHEDRON : { 0: { 0: 1.0 ,
                                     1: 1.0 ,
                                     2: 1.0 ,
                                     3: 1.0 } , \
                                1: { 0: rt2 , \
                                     1: rt2 , \
                                     2: rt2 , \
                                     3: 1.0 , \
                                     4: 1.0 , \
                                     5: 1.0 } , \
                                2: { 0: rt3 , \
                                     1: 1.0 , \
                                     2: 1.0 , \
                                     3: 1.0 } ,
                                3: { 0: 1.0 } } }
                                

# dictionary of shapes -- for each shape, maps edge number to normal
# which is a tuple of floats.
normals = { LINE : { 0: (-1.0,) , 1:(1.0,) } , \
            TRIANGLE : { 0: (rt2_inv,rt2_inv) , \
                         1: (-1.0,0) , \
                         2: (0,-1.0) } , \
            TETRAHEDRON : { 0: (rt3_inv,rt3_inv,rt3_inv) , \
                            1: (-1.0,0.0,0.0) , \
                            2: (0.0,-1.0,0.0) , \
                            3: (0.0,0.0,-1.0) } }

tangents = { TETRAHEDRON : { 0: (-rt2_inv,rt2_inv,0.0) , \
                             1: (0.0,-rt2_inv,rt2_inv) , \
                             2: (rt2_inv,0.0,-rt2_inv) , \
                             3: (0.0,0.0,-1.0) , \
                             4: (0.0,1.0,0.0) ,\
                             5: (1.0,0.0,0.0) } }

def scale_factor( shape , d , ent_id ):
    global jac_factors
    return jac_factors[ shape ][ d ][ ent_id ]

def dimension( shape ):
    """returns the topological dimension associated with shape."""
    global dims
    try:
        return dims[ shape ]
    except:
        raise ShapeError, "Illegal shape: shapes.dimension"

def dimension_range( shape ):
    """returns the list starting at zero and ending with
    the topological dimension of shape (inclusive).
    Hence, dimension_range( s ) is syntactic sugar for
    range(0,dimension(shape)+1)."""
    return range(0,dimension(shape)+1);

def entity_range( shape , dim ):
    """Returns the range of topological entities of dimension dim
    associated with shape.
    For example, entity_range( LINE , 0 ) returns the list [0,1]
    because there are two points associated with the line."""
    global num_entities
    try:
        return range(0,num_entities[ shape ][ dim ])
    except:
        raise ShapeError, "Illegal shape or dimension"

def polynomial_dimension( shape , n ):
    """Returns the number of polynomials of total degree n on the
    shape n.  This (n+1) over the line, (n+1)(n+2)/2 over the
    triangle, and (n+1)(n+2)(n+3)/6 in three dimensions."""
    d = dimension( shape )
    td = 1
    for i in xrange(0,d):
        td = td * ( n + i + 1 )
    return td / factorial( d )

