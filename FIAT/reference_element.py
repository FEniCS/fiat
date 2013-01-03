# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
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

"""
Abstract class and particular implementations of finite element
reference simplex geometry/topology.

Provides an abstract base class and particular implementations for the
reference simplex geometry and topology.
The rest of FIAT is abstracted over this module so that different
reference element geometry (e.g. a vertex at (0,0) versus at (-1,-1))
and orderings of entities have a single point of entry.

Currently implemented are UFC and Default Line, Triangle and Tetrahedron.
"""

import numpy

LINE = 1
TRIANGLE = 2
TETRAHEDRON = 3

def linalg_subspace_intersection( A , B ):
    """Computes the intersection of the subspaces spanned by the
    columns of 2-dimensional arrays A,B using the algorithm found in
    Golub and van Loan (3rd ed) p. 604.  A should be in
    R^{m,p} and B should be in R^{m,q}.  Returns an orthonormal basis
    for the intersection of the spaces, stored in the columns of
    the result."""

    # check that vectors are in same space
    if A.shape[0] != B.shape[0]:
        raise Exception("Dimension error")

    #A,B are matrices of column vectors
    # compute the intersection of span(A) and span(B)

    # Compute the principal vectors/angles between the subspaces, G&vL
    # p.604
    (qa,ra) = numpy.linalg.qr( A )
    (qb,rb) = numpy.linalg.qr( B )

    C = numpy.dot( numpy.transpose( qa ) , qb )

    (y,c,zt) = numpy.linalg.svd( C )

    U = numpy.dot( qa , y )
    V = numpy.dot( qb , numpy.transpose( zt ) )

    rank_c = len( [ s for s in c if numpy.abs( 1.0 - s ) < 1.e-10 ] )

    return U[:,:rank_c]

def lattice_iter( start , finish , depth ):
    """Generator iterating over the depth-dimensional lattice of
    integers between start and (finish-1).  This works on simplices in
    1d, 2d, 3d, and beyond"""
    if depth == 0:
        return
    elif depth == 1:
        for ii in range( start , finish ):
            yield [ii]
    else:
        for ii in range( start , finish ):
            for jj in lattice_iter( start , finish-ii , depth - 1 ):
                yield [ii] + jj

class ReferenceElement:
    """Abstract class for a reference element simplex.  Provides
    accessors for geometry (vertex coordinates) as well as topology
    (orderings of vertices that make up edges, facecs, etc."""
    def __init__( self , shape , vertices , topology ):
        """The constructor takes a shape code,
        the physical vertices expressed as a list of tuples
        of numbers, and the topology of a simplex.
        The topology is stored as a dictionary of dictionaries
        t[i][j] where i is the spatial dimension and j is the
        index of the facet of that dimension.  The result is
        a list of the vertices comprising the facet.
        """
        self.shape = shape
        self.vertices = vertices
        self.topology = topology
    def get_shape( self ):
        """Returns the code for the element's shape."""
        return self.shape
    def get_vertices( self ):
        """Returns an iteratble of the element's vertices, each stored
        as a tuple."""
        return self.vertices
    def get_spatial_dimension( self ):
        """Returns the spatial dimension in which the element lives."""
        return len( self.vertices[ 0 ] )
    def get_topology( self ):
        """Returns a dictionary encoding the topology of the element.
        The dictionary's keys are the spatial dimensions (0,1,...)
        and each value is a dictionary mapping
        """
        return self.topology
    def get_vertices_of_subcomplex( self , t ):
        """Returns the tuple of vertex coordinates associated with the
        labels contained in the iterable t."""
        return tuple( [ self.vertices[ ti ] for ti in t ] )
    def compute_normal( self , facet_i ):
        """Returns the unit normal vector to facet i of codimension 1."""
        # first, let's compute the span of the simplex
        # This is trivial if we have a d-simplex in R^d.
        # Not so otherwise.
        vert_vecs =  [ numpy.array( v ) \
                       for v in self.vertices ]
        vert_vecs_foo = numpy.array( [ vert_vecs[i] - vert_vecs[0] \
                                       for i in range(1,len(vert_vecs) ) ] )

        (u,s,vt) = numpy.linalg.svd( vert_vecs_foo )
        rank = len( [ si for si in s if si > 1.e-10 ] )

        # this is the set of vectors that span the simplex
        spanu = u[:,:rank]

        t = self.get_topology( )
        sd = self.get_spatial_dimension()
        vert_coords_of_facet = \
            self.get_vertices_of_subcomplex( t[sd-1][facet_i] )

        # now I find everything normal to the facet.
        vcf = [ numpy.array( foo ) \
                for foo in vert_coords_of_facet ]
        facet_span = numpy.array( [ vcf[i] - vcf[0] \
                                    for i in range(1,len(vcf) ) ] )
        (uf,sf,vft) = numpy.linalg.svd( facet_span )

        # now get the null space from vft
        rankfacet = len( [ si for si in sf if si > 1.e-10 ] )
        facet_normal_space = numpy.transpose( vft[rankfacet:,:] )

        # now, I have to compute the intersection of
        # facet_span with facet_normal_space
        foo = linalg_subspace_intersection( facet_normal_space , spanu )

        num_cols = foo.shape[1]

        if num_cols != 1:
            raise Exception("barf in normal computation")

        # now need to get the correct sign
        # get a vector in the direction
        nfoo = foo[:,0]

        # what is the vertex not in the facet?
        verts_set = set( t[sd][0] )
        verts_facet = set( t[sd-1][facet_i] )
        verts_diff = verts_set.difference( verts_facet )
        if len( verts_diff ) != 1:
            raise Exception("barf in normal computation: getting sign")
        vert_off = verts_diff.pop()
        vert_on = verts_facet.pop()

        # get a vector from the off vertex to the facet
        v_to_facet = numpy.array( self.vertices[vert_on] ) \
            - numpy.array( self.vertices[ vert_off ] )

        if numpy.dot( v_to_facet , nfoo ) > 0.0:
            return nfoo
        else:
            return -nfoo

    def compute_tangents( self , dim , i ):
        """computes tangents in any dimension based on differences
        between vertices and the first vertex of the i:th facet
        of dimension dim.  Returns a (possibly empty) list.
        These tangents are *NOT* normalized to have unit length."""
        t = self.get_topology()
        vs = list(map( numpy.array , \
                  self.get_vertices_of_subcomplex( t[dim][i] ) ))
        ts = [ v - vs[0] for v in vs[1:] ]
        return ts

    def compute_normalized_tangents( self , dim , i ):
        """computes tangents in any dimension based on differences
        between vertices and the first vertex of the i:th facet
        of dimension dim.  Returns a (possibly empty) list.
        These tangents are normalized to have unit length."""
        ts = self.compute_tangents( dim , i )
        return [ t / numpy.linalg.norm( t ) for t in ts ]

    def compute_edge_tangent( self , edge_i ):
        """Computes the nonnormalized tangent to a 1-dimensional facet.
        returns a single vector."""
        t = self.get_topology()
        (v0,v1) = self.get_vertices_of_subcomplex( t[1][edge_i] )
        return numpy.array( v1 ) - numpy.array( v0 )

    def compute_normalized_edge_tangent( self , edge_i ):
        """Computes the unit tangent vector to a 1-dimensional facet"""
        v = self.compute_edge_tangent( edge_i )
        return v / numpy.linalg.norm( v )

    def compute_face_tangents( self , face_i ):
        """Computes the two tangents to a face.  Only implemented
        for a tetrahedron."""
        if self.get_spatial_dimension() != 3:
            raise Exception("can't get face tangents yet")
        t = self.get_topology()
        (v0,v1,v2) = list(map( numpy.array , \
                          self.get_vertices_of_subcomplex( t[2][face_i] ) ))
        return (v1-v0,v2-v0)

    def make_lattice( self , n , interior = 0):
        """Constructs a lattice of points on the simplex.  For
        example, the 1:st order lattice will be just the vertices.
        The optional argument interior specifies how many points from
        the boundary to omit.  For example, on a line with n = 2,
        and interior = 0, this function will return the vertices and
        midpoint, but with interior = 1, it will only return the
        midpoint."""
        verts = self.get_vertices()
        nverts = len( verts )
        vs = [ numpy.array( v ) for v in verts ]
        hs = [ (vs[ i ] - vs[ 0 ]) / n for i in range(1,nverts) ]

        result = []

        m = len( hs )

        for indices in lattice_iter( interior , n + 1 - interior , m ):
            res_cur = vs[0].copy()
            for i in range(len(indices)):
                res_cur += indices[i] * hs[m-i-1]
            result.append( tuple( res_cur ) )

        return result

    def make_points( self , dim , entity_id , order ):
        """Constructs a lattice of points on the entity_id:th
        facet of dimension dim.  Order indicates how many points to
        include in each direction."""
        if dim == 0:
            return ( self.get_vertices()[entity_id] , )
        elif dim > self.get_spatial_dimension():
            raise Exception("illegal dimension")
        elif dim == self.get_spatial_dimension():
            return self.make_lattice( order , 1 )
        else:
            base_el = default_simplex( dim )
            base_verts = base_el.get_vertices()
            facet_verts = \
                        self.get_vertices_of_subcomplex( \
                            self.get_topology()[dim][entity_id] )

            (A,b) = make_affine_mapping( base_verts , facet_verts )

            f = lambda x: (numpy.dot( A , x ) + b)
            base_pts = base_el.make_lattice( order , 1 )
            image_pts = tuple( [ tuple( f( x ) ) for x in base_pts ] )

            return image_pts

    def volume( self ):
        """Computes the volumne of the simplex in the appropriate
        dimensional measure."""
        return volume( self.get_vertices() )

    def volume_of_subcomplex( self , dim , facet_no ):
        vids = self.topology[dim][facet_no]
        return volume( self.get_vertices_of_subcomplex( vids ) )

    def compute_scaled_normal( self , facet_i ):
        """Returns the unit normal to facet_i of scaled by the
        volume of that facet."""
        t = self.get_topology()
        sd = self.get_spatial_dimension()
        facet_verts_ids = t[sd-1][facet_i]
        facet_verts_coords = self.get_vertices_of_subcomplex( facet_verts_ids )

        v = volume( facet_verts_coords )

        return self.compute_normal( facet_i ) * v

class DefaultLine( ReferenceElement ):
    """This is the reference line with vertices (-1.0,) and (1.0,)."""
    def __init__( self ):
        verts = ( (-1.0,) , (1.0,) )
        edges = { 0 : ( 0 , 1 ) }
        topology = { 0 : { 0 : (0,) , 1: (1,) } , \
                     1 : edges }
        ReferenceElement.__init__( self , LINE , verts , topology )

class UFCInterval( ReferenceElement ):
    """This is the reference interval with vertices (0.0,) and (1.0,)."""
    def __init__( self ):
        verts = ( (0.0,) , (1.0,) )
        edges = { 0 : ( 0 , 1 ) }
        topology = { 0 : { 0 : (0,) , 1 : (1,) } , \
                     1 : edges }
        ReferenceElement.__init__( self , LINE , verts , topology )

class DefaultTriangle( ReferenceElement ):
    """This is the reference triangle with vertices (-1.0,-1.0),
    (1.0,-1.0), and (-1.0,1.0)."""
    def __init__( self ):
        verts = ((-1.0,-1.0),(1.0,-1.0),(-1.0,1.0))
        edges = { 0 : ( 1 , 2 ) , \
                  1 : ( 2 , 0 ) , \
                  2 : ( 0 , 1 ) }
        faces = { 0 : ( 0 , 1 , 2 ) }
        topology = { 0 : { 0 : (0,) , 1 : (1,) , 2 : (2,) } , \
                     1 : edges , 2 : faces }
        ReferenceElement.__init__( self , TRIANGLE , verts , topology )

class UFCTriangle( ReferenceElement ):
    """This is the reference triangle with vertices (0.0,0.0),
    (1.0,0.0), and (0.0,1.0)."""
    def __init__( self ):
        verts = ((0.0,0.0),(1.0,0.0),(0.0,1.0))
        edges = { 0 : ( 1 , 2 ) , 1 : ( 0 , 2 ) , 2 : ( 0 , 1 ) }
        faces = { 0 : ( 0 , 1 , 2 ) }
        topology = { 0 : { 0 : (0,) , 1 : (1,) , 2 : (2,) } , \
                     1 : edges , 2 : faces }
        ReferenceElement.__init__( self , TRIANGLE , verts , topology )

    def compute_normal(self, i):
        "UFC consistent normal"
        t = self.compute_tangents(1, i)[0]
        n = numpy.array((t[1], -t[0]))
        return n/numpy.linalg.norm(n)


class IntrepidTriangle( ReferenceElement ):
    """This is the Intrepid triangle with vertices (0,0),(1,0),(0,1)"""
    def __init__( self ):
        verts = ((0.0,0.0),(1.0,0.0),(0.0,1.0))
        edges = { 0 : ( 0 , 1 ) , \
                  1 : ( 1 , 2 ) , \
                  2 : ( 2 , 0 ) }
        faces = { 0 : ( 0 , 1 , 2 ) }
        topology = { 0 : { 0 : (0,) , 1 : (1,) , 2 : (2,) } , \
                     1 : edges , 2 : faces }
        ReferenceElement.__init__( self , TRIANGLE , verts , topology )


class DefaultTetrahedron( ReferenceElement ):
    """This is the reference tetrahedron with vertices (-1,-1,-1),
    (1,-1,-1),(-1,1,-1), and (-1,-1,1)."""
    def __init__( self ):
        verts = ((-1.0,-1.0,-1.0),(1.0,-1.0,-1.0),\
                 (-1.0,1.0,-1.0),(-1.0,-1.0,1.0))
        vs = { 0 : ( 0, ) , \
               1 : ( 1, ) , \
               2 : ( 2, ) , \
               3 : ( 3, ) }
        edges = { 0: ( 1 , 2 ) , \
                  1: ( 2 , 0 ) , \
                  2: ( 0 , 1 ) , \
                  3: ( 0 , 3 ) , \
                  4: ( 1 , 3 ) , \
                  5: ( 2 , 3 ) }
        faces = { 0 : ( 1 , 3 , 2 ) , \
                  1 : ( 2 , 3 , 0 ) , \
                  2 : ( 3 , 1 , 0 ) , \
                  3 : ( 0 , 1 , 2 ) }
        tets = { 0 : ( 0 , 1 , 2 , 3 ) }
        topology = { 0: vs , 1 : edges , 2 : faces , 3 : tets }
        ReferenceElement.__init__( self , TETRAHEDRON , verts , topology )

class IntrepidTetrahedron( ReferenceElement ):
    """This is the reference tetrahedron with vertices (0,0,0),
    (1,0,0),(0,1,0), and (0,0,1) used in the Intrepid project."""
    def __init__( self ):
        verts = ((0.0,0.0,0.0),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))
        vs = { 0 : ( 0, ) , \
               1 : ( 1, ) , \
               2 : ( 2, ) , \
               3 : ( 3, ) }
        edges = { 0 : (0,1) , \
                  1 : (1,2) , \
                  2 : (2,0) , \
                  3 : (0,3) , \
                  4 : (1,3) , \
                  5 : (2,3) }
        faces = { 0 : (0,1,3) , \
                  1 : (1,2,3) , \
                  2 : (0,3,2) , \
                  3 : (0,2,1) }
        tets = { 0 : ( 0 , 1 , 2 , 3 ) }
        topology = { 0: vs , 1 : edges , 2 : faces , 3 : tets }
        ReferenceElement.__init__( self , TETRAHEDRON , verts , topology )


class UFCTetrahedron( ReferenceElement ):
    """This is the reference tetrahedron with vertices (0,0,0),
    (1,0,0),(0,1,0), and (0,0,1)."""
    def __init__( self ):
        verts = ((0.0,0.0,0.0),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))
        vs = { 0 : ( 0, ) , \
               1 : ( 1, ) , \
               2 : ( 2, ) , \
               3 : ( 3, ) }
        edges = { 0 : ( 2 , 3 ) , \
                  1 : ( 1 , 3 ) , \
                  2 : ( 1 , 2 ) , \
                  3 : ( 0 , 3 ) , \
                  4 : ( 0 , 2 ) , \
                  5 : ( 0 , 1 ) }
        faces = { 0 : ( 1 , 2 , 3 ) , \
                  1 : ( 0 , 2 , 3 ) , \
                  2 : ( 0 , 1 , 3 ) , \
                  3 : ( 0 , 1 , 2 ) }
        tets = { 0 : ( 0 , 1 , 2 , 3 ) }
        topology = { 0: vs , 1 : edges , 2 : faces , 3 : tets }
        ReferenceElement.__init__( self , TETRAHEDRON , verts , topology )

    def compute_normal(self, i):
        "UFC consistent normals."
        t = self.compute_tangents(2, i)
        n = numpy.cross(t[0], t[1])
        return -2.0*n/numpy.linalg.norm(n)


def make_affine_mapping( xs , ys ):
    """Constructs (A,b) such that x --> A * x + b is the affine
    mapping from the simplex defined by xs to the simplex defined by ys."""

    dim_x = len( xs[0] )
    dim_y = len( ys[0] )

    if len( xs ) != len( ys ):
        raise Exception("")

    # find A in R^{dim_y,dim_x}, b in R^{dim_y} such that
    # A xs[i] + b = ys[i] for all i

    mat = numpy.zeros( (dim_x*dim_y+dim_y,dim_x*dim_y+dim_y) , "d" )
    rhs = numpy.zeros( (dim_x*dim_y+dim_y,) , "d" )

    # loop over points
    for i in range( len( xs ) ):
        # loop over components of each A * point + b
        for j in range( dim_y ):
            row_cur = i*dim_y+j
            col_start = dim_x * j
            col_finish = col_start + dim_x
            mat[row_cur,col_start:col_finish] = numpy.array( xs[i] )
            rhs[row_cur] = ys[i][j]
            # need to get terms related to b
            mat[row_cur,dim_y*dim_x+j] = 1.0

    sol = numpy.linalg.solve( mat , rhs )

    A = numpy.reshape( sol[:dim_x*dim_y] , (dim_y,dim_x) )
    b = sol[dim_x*dim_y:]

    return A,b



def default_simplex( spatial_dim ):
    """Factory function that maps spatial dimension to an instance of
    the default reference simplex of that dimension."""
    if spatial_dim == 1:
        return DefaultLine()
    elif spatial_dim == 2:
        return DefaultTriangle()
    elif spatial_dim == 3:
        return DefaultTetrahedron()

def ufc_simplex( spatial_dim ):
    """Factory function that maps spatial dimension to an instance of
    the UFC reference simplex of that dimension."""
    if spatial_dim == 1:
        return UFCInterval()
    elif spatial_dim == 2:
        return UFCTriangle()
    elif spatial_dim == 3:
        return UFCTetrahedron()
    else:
        raise RuntimeError("Don't know how to create UFC simplex for dimension %s" % str(spatial_dim))

def volume( verts ):
    """Constructs the volume of the simplex spanned by verts"""
    from .factorial import factorial
    # use fact that volume of UFC reference element is 1/n!
    sd = len( verts ) - 1
    ufcel = ufc_simplex( sd )
    ufcverts = ufcel.get_vertices()

    A,b = make_affine_mapping( ufcverts , verts )

    # can't just take determinant since, e.g. the face of
    # a tet being mapped to a 2d triangle doesn't have a
    # square matrix

    (u,s,vt) = numpy.linalg.svd( A )

    # this is the determinant of the "square part" of the matrix
    # (ie the part that maps the restriction of the higher-dimensional
    # stuff to UFC element
    p = numpy.prod( [ si for si in s if (si) > 1.e-10 ] )

    return p / factorial( sd )

if __name__ == "__main__":
#    U = UFCTetrahedron()
#    print U.make_points( 1 , 1 , 3 )
#    for i in range(len(U.vertices)):
#        print U.compute_normal( i )

    V = DefaultTetrahedron()
    sd = V.get_spatial_dimension()

#    print make_affine_mapping(V.get_vertices(),U.get_vertices())

    for i in range( len( V.vertices ) ):
        print(V.compute_normal( i ))
        print(V.compute_scaled_normal( i ))
        print(volume( V.get_vertices_of_subcomplex( V.topology[sd-1][i] ) ))
        print()
