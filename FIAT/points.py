# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Last modified 6 April 2005

"""points.py gives locations for edge, face, tetrahedron points
for each simplex shape.
The major functions to use are
make_lattice( shape , n )
and
make_points( shape , dim , entity_id , order )
These functions are used to put down a regular lattice over a simplex
or to extract the points associated with particular entities of a
particular topological dimension in the lattice of a particular
order.  This is very useful in determining nodal locations for finite
element dual bases."""

import shapes
from curry import curry
from math import sqrt


rt2 = sqrt(2.0)

# these three functions are called within the function
# make_lattice (below)
def make_lattice_line(n):
    return tuple( [ ( -1.0 + (2.0 * jj) / n , )\
                    for jj in range(0,n+1) ] )

def make_lattice_triangle(n):
    return tuple( [ ( -1.0 + (2.0 * jj) / n  , -1.0 + (2.0 * ii) / n ) \
		    for ii in xrange(0,n+1) \
		    for jj in xrange(0,n-ii+1)] )

def make_lattice_tetrahedron( n ):
    return tuple( [ (-1.+(2.*jj)/n,-1.+(2.*ii)/n,-1.+(2.*kk)/n) \
                    for ii in xrange(0,n+1) \
                    for jj in xrange(0,n-ii+1) \
                    for kk in xrange(0,n-ii-jj+1) ] )

lattice_funcs = { shapes.LINE: make_lattice_line , \
                  shapes.TRIANGLE: make_lattice_triangle, \
                  shapes.TETRAHEDRON: make_lattice_tetrahedron }

def make_lattice( shape , n ):
    global lattice_funcs
    try:
        return lattice_funcs[ shape ]( n )
    except:
        raise shapes.ShapeError, "Illegal shape: points.make_lattice"


# Functions that map a single point on the line to an edge
# on a triangle
pt_to_edge_triangle = { 0:lambda x: (-x[0],x[0]), \
                        1:lambda x: (-1.0,-x[0]), \
                        2:lambda x: (x[0],-1.0) }


tuple_pt_to_edge_triangle = { 0:lambda x: (-x[0],x[0]), \
                        1:lambda x: (-1.0,-x[0]), \
                        2:lambda x: (x[0],-1.0) }

tuple_pt_to_edge_tetrahedron = { 0: lambda x: (-x[0],x[0],-1.),   \
                           1: lambda x: (-1.,-x[0],x[0]),   \
                           2: lambda x: (x[0],-1.,-x[0]),   \
                           3: lambda x: (-1.,-1.,-x[0]), \
                           4: lambda x: (-1.,x[0],-1.),  \
                           5: lambda x: (x[0],-1.,-1.) }


tuple_pt_to_edge = { shapes.TRIANGLE: tuple_pt_to_edge_triangle , \
                     shapes.TETRAHEDRON: tuple_pt_to_edge_tetrahedron }

# Functions that map a single point on the line to an edge
# on a tet
pt_to_edge_tetrahedron = { 0: lambda x: (-x[0],x[0],-1.),   \
                           1: lambda x: (-1.,-x[0],x[0]),   \
                           2: lambda x: (x[0],-1.,-x[0]),   \
                           3: lambda x: (-1.,-1.,-x[0]), \
                           4: lambda x: (-1.,x[0],-1.),  \
                           5: lambda x: (x[0],-1.,-1.) }

# functions that take a point in the plane (ref triangle) and
# map it to the appropriate face of the ref tet
pt_to_face_tetrahedron = { 0: lambda x: (-x[0]-x[1]-1.,x[0],x[1]),
                           1: lambda x: (-1.,-x[0]-x[1]-1.,x[0]),
                           2: lambda x: (x[1],-1.,-x[0]-x[1]-1.), \
                           3: lambda x: (x[0],x[1],-1.) }

# dictionaries that dispatch on shape type
pt_to_edge = { shapes.TRIANGLE: pt_to_edge_triangle , \
	       shapes.TETRAHEDRON: pt_to_edge_tetrahedron }

pt_to_face = { shapes.TETRAHEDRON: pt_to_face_tetrahedron }

pt_maps = { shapes.LINE : { } , \
            shapes.TRIANGLE: { 1 : pt_to_edge_triangle } , \
            shapes.TETRAHEDRON: { 1 : pt_to_edge_tetrahedron , \
                                  2 : pt_to_face_tetrahedron } }



# functions that get points of a particular order
# on a mesh component of particular dimension
# these are not needed in the public interface,
# which is simply the function make_points.

def make_vertex_points_line( vid , order ):
    try:
        return ( shapes.vertices[shapes.LINE][vid] , )
    except:
        raise RuntimeError, "Illegal vertex number"

def make_vertex_points_triangle( vid , order ):
    global vertices
    try:
        return ( shapes.vertices[shapes.TRIANGLE][vid], )
    except:
        raise RuntimeError, "Illegal vertex number"

def make_vertex_points_tetrahedron( vid , order ):
    global vertices
    try:
        return ( shapes.vertices[shapes.TETRAHEDRON][vid], )
    except:
        raise RuntimeError, "Illegal vertex number"

def make_edge_points_line( eid , order ):
    if eid != 0:
        raise RuntimeError, "Illegal edge number"
    else:
        return tuple( map(lambda x: (x,) ,line_range(order)) )

def make_edge_points_triangle( eid , order ):
    global pt_to_edge_triangle
    try:
        return tuple( [ pt_to_edge_triangle[eid]( x ) \
                        for x in line_range( order ) ] )
    except:
        raise RuntimeError, "Illegal edge number."

def make_edge_points_tetrahedron( eid , order ):
    global pt_to_edge_tetrahedron
    try:
        return tuple( [ pt_to_edge_tetrahedron[eid]( x ) \
                        for x in line_range( order ) ] )
    except:
        raise RuntimeError, "Illegal edge number"

def make_face_points_tetrahedron( fid , order ):
    try:
        return tuple( [ pt_to_face_tetrahedron[fid]( x ) \
                        for x in \
                        make_interior_points_triangle( 0 , order ) ] )
    except:
        raise RuntimeError, "Can't make triangle points"


def make_interior_points_triangle( iid , order ):
    if iid != 0:
        raise RuntimeError, "Illegal id"
    return tuple( [ ( -1.0 + (2.0 * jj) / order  , \
                      -1.0 + (2.0 * ii) / order ) \
                    for ii in range(1,order-1) \
                    for jj in range(1,order-ii)] )

def make_interior_points_tetrahedron( iid , n ):
    if iid != 0:
        raise RuntimeError, "Illegal id"
    return tuple( [ (-1.+(2.*jj)/n,-1.+(2.*ii)/n,-1.+(2.*kk)/n) \
                    for ii in xrange(1,n) \
                    for jj in xrange(1,n-ii) \
                    for kk in xrange(1,n-ii-jj) ] )


def line_range( n ):
    f = curry( lambda m,i : (-1.0 + (2.0*i)/m,) , n )
    return tuple( map( f , range(1,n) ) )


#def line_range(n):
#    f = curry( lambda m,i : -1.0 + (2.0*i)/m , n )
#    return tuple( map( f , range(1,n) ) )

# dictionaries mapping dimension of mesh entities to
# functions for generating points of a particular order
# on those mesh entities

line_pts = { 0: make_vertex_points_line , \
             1: make_edge_points_line }

tri_pts = { 0: make_vertex_points_triangle , \
            1: make_edge_points_triangle , \
            2: make_interior_points_triangle }

tet_pts = { 0: make_vertex_points_tetrahedron , \
            1: make_edge_points_tetrahedron , \
            2: make_face_points_tetrahedron , \
            3: make_interior_points_tetrahedron }

pt_funcs = { shapes.LINE: line_pts , \
             shapes.TRIANGLE: tri_pts , \
             shapes.TETRAHEDRON: tet_pts }

def make_points( shape , dim , entity_id , order ):
    """Main public interface for getting points associated with mesh
    components on reference domain.  You give it a shape
    ( defined in shapes.py) the topological dimension of the
    entity (0 for vertices, 1 for lines, etc), the id of the mesh
    entity of that dimension, and the order of points, and it gives
    you the appropriate tuple."""
    global pt_funcs
    try:
        return pt_funcs[ shape ][ dim ]( entity_id , order )
    except:
        raise RuntimeError, "Can't make those points"


def barycentric_to_ref_tri( pt_bary ):
    (L1,L2,L3) = pt_bary
    x = -L1 + L2 - L3
    y = -L1 - L2 + L3
    return (x,y)

