# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 1 Jun 2005 by RCK

"""shapes.py
Module defining topological and geometric information for simplices in one, two, and
three dimensions."""
import Numeric, LinearAlgebra, exceptions
from curry import curry
from factorial import factorial
from math import sqrt


def strike_col( A , j ):
    m,n = A.shape
    return Numeric.take( A , [ k for k in range(0,n) if k != j ] , 1 )

def cross( vecs ):
    """Multidimensional cross product of d+1 vecs in R^{d}."""
    n,d = len(vecs),len(vecs[0]) 
    mat = Numeric.array( vecs )
    if n != d-1:
        raise RuntimeError, "This won't work"
    return Numeric.array( \
        [ (-1)**i * LinearAlgebra.determinant(strike_col(mat,i)) \
          for i in range(0,d) ] )

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
dim = { LINE: 1 , \
         TRIANGLE: 2 , \
         TETRAHEDRON: 3 }

# number of topological entities of each dimensions
# associated with each shape.
# 0 refers to the number of points that make up the simplex
# 1 refers to the number of edges
# 2 refers to the number of triangles/faces
# 3 refers to the number of tetrahedra
num_entities = { LINE: { 0: 2 , 1: 1 } ,\
                 TRIANGLE: { 0: 3, 1: 3, 2:1 } , \
                 TETRAHEDRON: {0: 4, 1: 6, 2: 4, 3: 1} }

# canonical polynomial dimension associated with simplex
# of each dimension
poly_dims = { LINE: lambda n: max(0,n + 1), \
              TRIANGLE: lambda n: max(0,(n+1)*(n+2)/2) ,\
              TETRAHEDRON: lambda n: max(0,(n+1)*(n+2)*(n+3)/6) }

# dictionaries of vertices.  I'm switching to (0,1) rather than (-1,1)
# This means I need a wrapper around the expansion functions
vertices = { LINE : { 0 : ( 0.0 , ) , 1 : ( 1.0 , ) } , \
             TRIANGLE : { 0 : ( 0.0 , 0.0 ) , \
                          1 : ( 1.0 , 0.0 ) , \
                          2 : ( 0.0 , 1.0 ) } , \
             TETRAHEDRON : { 0 : ( 0.0 , 0.0 , 0.0 ) , \
                             1 : ( 1.0 , 0.0 , 0.0 ) , \
                             2 : ( 0.0 , 1.0 , 0.0 ) , \
                             3 : ( 0.0 , 0.0 , 1.0 ) } }

def distance( a , b ):
    if len( a ) != len( b ):
        raise ShapeError, "Can't compute the distance"
    d = 0.0
    for i in range( len( a ) ):
        d += ( b[i] - a[i] ) ** 2
    return sqrt( d )

def area( a , b , c ):
    if len( a ) != 3 or len( b ) != 3 or len( c ) != 3:
        raise ShapeError, "Can't compute area"
    v1 = Numeric.array( a ) - Numeric.array( c )
    v2 = Numeric.array( b ) - Numeric.array( c )
    crss = cross( [ v1 , v2 ] )
    return sqrt( Numeric.dot( crss , crss ) )

# mapping from edge ids of a triangle to the pair of vertices
triangle_edges = { 0 : ( 1 , 2 ) , \
                   1 : ( 2 , 0 ) , \
                   2 : ( 0 , 1 ) }

        
tetrahedron_edges = { 0 : ( 1 , 2 ) , \
                     1 : ( 2 , 0 ) , \
                     2 : ( 0 , 1 ) , \
                     3 : ( 0 , 3 ) , \
                     4 : ( 1 , 3 ) , \
                     5 : ( 2 , 3 ) }


tetrahedron_faces = { 0 : ( 1 , 2 , 3 ) , \
                      1 : ( 2 , 3 , 0 ) , \
                      2 : ( 3 , 0 , 1 ) , \
                      3 : ( 0 , 1 , 2) }

edges = { TRIANGLE : triangle_edges , \
          TETRAHEDRON : tetrahedron_edges }

faces = { TETRAHEDRON : tetrahedron_faces }


vertex_relation = { LINE : { 1 : { 0 : tuple( range( 2 ) ) } } , \
                    TRIANGLE : { 1 : triangle_edges , \
                                 2 : { 0 : tuple( range( 3 ) ) } }, \
                    TETRAHEDRON : { 1 : tetrahedron_edges , \
                                    2 : tetrahedron_faces , \
                                    3 : { 0 : tuple( range( 4 ) ) } } }

edge_jac_factors = {}
for shp in ( TRIANGLE , TETRAHEDRON ):
    edge_jac_factors[ shp ] = {}
    for i in range( num_entities[ shp ][ 1 ] ):
        verts = edges[ shp ][ i ]
        a = vertices[ shp ][ verts[ 0 ] ]
        b = vertices[ shp ][ verts[ 1 ] ]
        edge_jac_factors[ shp ][ i ] = distance( a , b )

face_jac_factors = {}
for shp in ( TETRAHEDRON , ):
    face_jac_factors[ shp ] = {}
    for i in range( num_entities[ shp ][ 2 ] ):
        verts = faces[ shp ][ i ]
        v0 = vertices[ shp ][ verts[ 0 ] ]
        v1 = vertices[ shp ][ verts[ 1 ] ]
        v2 = vertices[ shp ][ verts[ 2 ] ]
        face_jac_factors[ shp ][ i ] = area( v0 , v1 , v2 )
        
jac_factors = { TRIANGLE : { 1 : edge_jac_factors[ TRIANGLE ] } , \
                TETRAHEDRON : { 1 : edge_jac_factors[ TETRAHEDRON ] , \
                                2 : face_jac_factors[ TETRAHEDRON ] } }

normals = {}
for shp in ( TRIANGLE , TETRAHEDRON ):
    normals[ shp ] = {}
    vert_dict = vertex_relation[ shp ][ dim[ shp ] - 1 ]
    for i in vert_dict.keys():
        vert_ids = vert_dict[ i ]
        vert_vecs = [ Numeric.array( vertices[ shp ][ j ] ) \
                      for j in vert_ids ]
        vecs = [ v - vert_vecs[ 0 ] for v in vert_vecs[ 1: ] ]
        crss = cross( vecs )
        normals[ shp ][ i ] = crss / sqrt( Numeric.dot( crss , crss ) )

tangents = {}
for shp in ( TRIANGLE , TETRAHEDRON ):
    tangents[ shp ] = {}
    vert_dict = vertex_relation[ shp ][ 1 ]
    for i in vert_dict.keys():
        vert_ids = vert_dict[ i ]
        vert_vecs = [ Numeric.array( vertices[ shp ][ j ] ) \
                      for j in vert_ids ]
        diff = vert_vecs[ 1 ] - vert_vecs[ 0 ]
        tangents[ shp ][ i ] = diff / sqrt( Numeric.dot( diff , diff ) )
    
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

def make_lattice_line( n , interior = 0 ):
    if n == 0:
        return tuple( )
    v0 = vertices[ LINE ][0][0]
    v1 = vertices[ LINE ][1][0]
    h = abs( v1 - v0 )
    return tuple( [ ( v0 + ( h * jj ) / n , ) \
                    for jj in xrange( interior , n + 1 - interior ) ] )

def make_lattice_triangle( n , interior = 0 ):
    if n == 0: return tuple( )
    vs = [ Numeric.array( vertices[ TRIANGLE ][ i ] ) \
             for i in vertices[ TRIANGLE ].iterkeys() ]
    hs = [ vs[ i ] - vs[ 0 ] for i in (1,2) ]
    return tuple( [ tuple( vs[ 0 ] + ii * hs[ 1 ] / n + jj * hs[ 0 ] / n ) \
                    for ii in xrange( interior , n + 1 - interior ) \
                        for jj in xrange( interior , n - ii + 1 - interior ) ] )

def make_lattice_tetrahedron( n , interior = 0 ):
    if n == 0: return tuple( )
    vs = [ Numeric.array( vertices[ TETRAHEDRON ][ i ] ) \
             for i in vertices[ TETRAHEDRON ].iterkeys() ]
    hs = [ vs[ i ] - vs[ 0 ] for i in (1,2,3) ]
    return tuple( [ tuple( vs[ 0 ] \
                           + ii * hs[ 2 ] / n \
                           + jj * hs[ 1 ] / n \
                           + kk * hs[ 0 ] / n ) \
                    for ii in xrange( interior , n + 1 - interior ) \
                        for jj in xrange( interior , \
                                          n - ii + 1 - interior ) \
                            for kk in xrange( interior , \
                                              n - ii - jj + 1 - interior ) ] )

lattice_funcs = { LINE : make_lattice_line , \
                  TRIANGLE : make_lattice_triangle , \
                  TETRAHEDRON : make_lattice_tetrahedron }

def make_lattice( shp , n , interior = 0 ):
    return lattice_funcs[ shp ]( n , interior )    

def make_pt_to_edge( shp , verts ):
    """verts is the tuple of vertex ids on the reference shape.
    Returns the function mapping points (1-tuples) in [0,1] to points in
    d-d (2- or 3- tuples) that are on that edge of the reference shape."""
    v0 = Numeric.array( vertices[ shp ][ verts[ 0 ] ] )
    v1 = Numeric.array( vertices[ shp ][ verts[ 1 ] ] )
    diff = v1 - v0
    return lambda x: tuple( v0 + x[ 0 ] * diff )

def make_pt_to_face_tetrahedron( verts ):
    """verts is a triple of vertex ids on the reference tetrahedron.
    Returns a function mapping points (2-tuples) on the reference triangle
    to points on that face of the reference tetrahedron."""
    v0 = Numeric.array( vertices[ TETRAHEDRON ][ verts[ 0 ] ] )
    v1 = Numeric.array( vertices[ TETRAHEDRON ][ verts[ 1 ] ] )
    v2 = Numeric.array( vertices[ TETRAHEDRON ][ verts[ 2 ] ] )
    d = [ v1 - v0 , v2 - v0 ]
    return lambda x: tuple( v0 + x[ 0 ] * d[ 0 ] + x[ 1 ] * d[ 1 ] )

pt_to_edge = { TRIANGLE : \
               lambda i: \
                   make_pt_to_edge( TRIANGLE , \
                                    edges[ TRIANGLE ][ i ] ) , \
               TETRAHEDRON : \
               lambda i: \
                   make_pt_to_edge( TETRAHEDRON , \
                                    edges[ TETRAHEDRON ][ i ] ) }

pt_to_face = { TETRAHEDRON : \
               lambda i: \
                   make_pt_to_face_tetrahedron( faces[ TETRAHEDRON ][ i ] ) }

pt_maps = { LINE : {} , \
            TRIANGLE : { 1 : pt_to_edge[ TRIANGLE ] } , \
            TETRAHEDRON : { 1 : pt_to_edge[ TETRAHEDRON ] ,
                            2 : pt_to_face[ TETRAHEDRON ] } }
                            
def make_vertex_points( shp , vid , order ):
    return vertices[ shp ][ vid ]

def make_edge_points( shp , eid , order ):
    f = pt_maps[ shp ][ 1 ]( eid )
    xs = make_lattice( LINE , order )[1:-1]
    return tuple( map( f , xs ) )

