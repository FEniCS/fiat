# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Modified 27 Sept 2005 by RCK to fix normal directions on triangles
# Modified 23 Sept 2005 by RCK
# -- was off by some constant factors in scale_factors
# Modified 15 Sept 2005 by RCK
# Modified 1 Jun 2005 by RCK

"""shapes.py
Module defining topological and geometric information for simplices in one, two, and
three dimensions."""
import numpy, numpy.linalg, exceptions
from factorial import factorial
from math import sqrt
import numbering, reference # UFC vs. FIAT modules.


def strike_col( A , j ):
    m,n = A.shape
    return numpy.take( A , [ k for k in range(0,n) if k != j ] , 1 )

def cross( vecs ):
    """Multidimensional cross product of d+1 vecs in R^{d}."""
    n,d = len(vecs),len(vecs[0]) 
    mat = numpy.array( vecs )
    if n != d-1:
        raise RuntimeError, "This won't work"
    return numpy.array( \
        [ (-1)**i * numpy.linalg.det(strike_col(mat,i)) \
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

# Entities dependent on the choice of reference element:
# Dictionaries of vertices and the scale of the reference element
vertices = reference.get_vertices()
scale = reference.get_scale()

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
    v1 = numpy.array( a ) - numpy.array( c )
    v2 = numpy.array( b ) - numpy.array( c )
    crss = cross( [ v1 , v2 ] )
    return sqrt( numpy.dot( crss , crss ) )

# mapping from edge ids of a triangle to the pair of vertices
(triangle_edges, tetrahedron_edges, tetrahedron_faces) = numbering.get_entities()

edges = { TRIANGLE : triangle_edges , \
          TETRAHEDRON : tetrahedron_edges }

faces = { TRIANGLE : { 0 : ( 0 , 1 , 2 ) } , \
          TETRAHEDRON : tetrahedron_faces }


vertex_relation = { LINE : { 1 : { 0 : tuple( range( 2 ) ) } } , \
                    TRIANGLE : { 1 : triangle_edges , \
                                 2 : { 0 : tuple( range( 3 ) ) } }, \
                    TETRAHEDRON : { 1 : tetrahedron_edges , \
                                    2 : tetrahedron_faces , \
                                    3 : { 0 : tuple( range( 4 ) ) } } }

edge_jac_factors = {}
for shp in ( TRIANGLE , TETRAHEDRON ):
    edge_jac_factors[ shp ] = {}
    num_edges = num_entities[ shp ][ 1 ]
    for i in range( num_edges ):
        verts = edges[ shp ][ i ]
        a = vertices[ shp ][ verts[ 0 ] ]
        b = vertices[ shp ][ verts[ 1 ] ]
        edge_jac_factors[ shp ][ i ] = distance( a , b )/scale

# hard-wired for reference element on [-1,1]
# meg: What about this is hard-wired?
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
shp=TRIANGLE
normals[shp]={}
vert_dict = vertex_relation[ shp ][ dim[ shp ] - 1 ]
for i in vert_dict.keys():
    vert_ids = vert_dict[ i ]
    vert_vecs = [ numpy.array( vertices[ shp ][ j ] ) \
                  for j in vert_ids ]
    vecs = [ v - vert_vecs[ 0 ] for v in vert_vecs[ 1: ] ]
    crss = cross( vecs )
    normals[ shp ][ i ] = crss / sqrt( numpy.dot( crss , crss ) )

shp=TETRAHEDRON
normals[ shp ] = {}
vert_dict = vertex_relation[ shp ][ dim[ shp ] - 1 ]
for i in vert_dict.keys():
    vert_ids = vert_dict[ i ]
    vert_vecs = [ numpy.array( vertices[ shp ][ j ] ) \
                  for j in vert_ids ]
    vecs = [ v - vert_vecs[ 0 ] for v in vert_vecs[ 1: ] ]
    crss = cross( vecs )
    normals[ shp ][ i ] = -crss / sqrt( numpy.dot( crss , crss ) )

tangents = {}
for shp in ( TRIANGLE , TETRAHEDRON ):
    tangents[ shp ] = {}
    tangents[shp][1] = {}
    vert_dict = vertex_relation[ shp ][ 1 ]
    for i in vert_dict.keys():
        vert_ids = vert_dict[ i ]
        vert_vecs = [ numpy.array( vertices[ shp ][ j ] ) \
                      for j in vert_ids ]
        diff = vert_vecs[ 1 ] - vert_vecs[ 0 ]
        tangents[ shp ][1][ i ] = diff / sqrt( numpy.dot( diff , diff ) )

tangents[TETRAHEDRON][2] = {}
for f in range(4):
    vert_ids = vertex_relation[ TETRAHEDRON ][ 2 ][ f ]
    v = numpy.array( [ vertices[ TETRAHEDRON ][ vert_ids[i ] ] for i in range(3) ] )
    v10 = v[1]-v[0]
    v20 = v[2]-v[0]
    t0 = v10 / numpy.sqrt( numpy.dot( v10 , v10 ) )
    alpha = sum( v20 * v10 ) / sum( v10 * v10 )
    t1nonunit = v[2] - v[0] - alpha * v[1]
    t1 = t1nonunit / numpy.sqrt( numpy.dot( t1nonunit , t1nonunit ) )
    tangents[TETRAHEDRON][2][ f ] = (t0,t1)
    
def scale_factor( shape , d , ent_id ):
    global jac_factors
    return jac_factors[ shape ][ d ][ ent_id ]

def dimension( shape ):
    """returns the topological dimension associated with shape."""
    global dim
    try:
        return dim[ shape ]
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
    vs = [ numpy.array( vertices[ TRIANGLE ][ i ] ) \
             for i in vertices[ TRIANGLE ].iterkeys() ]
    hs = [ vs[ i ] - vs[ 0 ] for i in (1,2) ]
    return tuple( [ tuple( vs[ 0 ] + ii * hs[ 1 ] / n + jj * hs[ 0 ] / n ) \
                    for ii in xrange( interior , n + 1 - interior ) \
                        for jj in xrange( interior , n - ii + 1 - interior ) ] )

def make_lattice_tetrahedron( n , interior = 0 ):
    if n == 0: return tuple( )
    vs = [ numpy.array( vertices[ TETRAHEDRON ][ i ] ) \
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
    a0 = vertices[ 1 ][ 0 ][ 0 ]
    b0 = vertices[ 1 ][ 1 ][ 0 ]
    h = 1./(b0 - a0)
    v0 = numpy.array( vertices[ shp ][ verts[ 0 ] ] )
    v1 = numpy.array( vertices[ shp ][ verts[ 1 ] ] )
    diff = v1 - v0
    return lambda x: tuple( v0 + h * ( x[ 0 ] - a0 ) * diff )

def make_pt_to_face( shp , verts ):
    if shp == TRIANGLE: return lambda x: x
    else: return make_pt_to_face_tetrahedron( verts )

def make_pt_to_face_tetrahedron( verts ):
    """verts is a triple of vertex ids on the reference tetrahedron.
    Returns a function mapping points (2-tuples) on the reference triangle
    to points on that face of the reference tetrahedron."""
    v0 = numpy.array( vertices[ TETRAHEDRON ][ verts[ 0 ] ] )
    v1 = numpy.array( vertices[ TETRAHEDRON ][ verts[ 1 ] ] )
    v2 = numpy.array( vertices[ TETRAHEDRON ][ verts[ 2 ] ] )
    
    c0 = 1.0/(distance(vertices[TRIANGLE][0], vertices[TRIANGLE][1]))
    c1 = 1.0/(distance(vertices[TRIANGLE][0], vertices[TRIANGLE][2]))
    a = numpy.array(vertices[TRIANGLE][0])
    d = [ v1 - v0 , v2 - v0 ]
    return lambda x: tuple( v0 \
                            + c0*(x[ 0 ] - a[ 0 ]) * d[ 0 ] \
                            + c1*(x[ 1 ] - a[ 1 ]) * d[ 1 ] )

pt_to_edge = { TRIANGLE : \
               lambda i: \
                   make_pt_to_edge( TRIANGLE , \
                                    edges[ TRIANGLE ][ i ] ) , \
               TETRAHEDRON : \
               lambda i: \
                   make_pt_to_edge( TETRAHEDRON , \
                                    edges[ TETRAHEDRON ][ i ] ) }

pt_to_face = { TRIANGLE : \
               lambda i: \
                   make_pt_to_face( TRIANGLE , faces[ TRIANGLE ][ i ] ) ,
               TETRAHEDRON : \
               lambda i: \
                   make_pt_to_face( TETRAHEDRON , faces[ TETRAHEDRON ][ i ] ) }

pt_maps = { LINE : { } , \
            TRIANGLE : { 1 : pt_to_edge[ TRIANGLE ] , \
                         2 : pt_to_face[ TRIANGLE ] } , \
            TETRAHEDRON : { 1 : pt_to_edge[ TETRAHEDRON ] ,
                            2 : pt_to_face[ TETRAHEDRON ] , \
                            3 : lambda y: lambda x: x } }
                            
def make_points( shp , dim , entity_id , order ):
    if dim == 0:
        return ( vertices[ shp ][ entity_id ] , )
    if dim == shp:
        if entity_id == 0:
            return make_lattice( shp , order , 1 )
        else:
            raise ShapeError, "Can't make those points"
    else:
        f = pt_maps[ shp ][ dim ]( entity_id )
        xs = make_lattice( dim , order , 1 )
        return tuple( map( f , xs ) )
    
