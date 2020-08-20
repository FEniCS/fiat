# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2014
# Modified by Lizao Li (lzlarryli@gmail.com), 2016

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
from itertools import chain, product, count
from functools import reduce
from collections import defaultdict
import operator
from math import factorial

import numpy


POINT = 0
LINE = 1
TRIANGLE = 2
TETRAHEDRON = 3
QUADRILATERAL = 11
HEXAHEDRON = 111
TENSORPRODUCT = 99


def lattice_iter(start, finish, depth):
    """Generator iterating over the depth-dimensional lattice of
    integers between start and (finish-1).  This works on simplices in
    1d, 2d, 3d, and beyond"""
    if depth == 0:
        return
    elif depth == 1:
        for ii in range(start, finish):
            yield [ii]
    else:
        for ii in range(start, finish):
            for jj in lattice_iter(start, finish - ii, depth - 1):
                yield jj + [ii]


def make_lattice(verts, n, interior=0):
    """Constructs a lattice of points on the simplex defined by verts.
    For example, the 1:st order lattice will be just the vertices.
    The optional argument interior specifies how many points from
    the boundary to omit.  For example, on a line with n = 2,
    and interior = 0, this function will return the vertices and
    midpoint, but with interior = 1, it will only return the
    midpoint."""

    vs = numpy.array(verts)
    hs = (vs - vs[0])[1:, :] / n

    m = hs.shape[0]
    result = [tuple(vs[0] + numpy.array(indices).dot(hs))
              for indices in lattice_iter(interior, n + 1 - interior, m)]

    return result


def linalg_subspace_intersection(A, B):
    """Computes the intersection of the subspaces spanned by the
    columns of 2-dimensional arrays A,B using the algorithm found in
    Golub and van Loan (3rd ed) p. 604.  A should be in
    R^{m,p} and B should be in R^{m,q}.  Returns an orthonormal basis
    for the intersection of the spaces, stored in the columns of
    the result."""

    # check that vectors are in same space
    if A.shape[0] != B.shape[0]:
        raise Exception("Dimension error")

    # A,B are matrices of column vectors
    # compute the intersection of span(A) and span(B)

    # Compute the principal vectors/angles between the subspaces, G&vL
    # p.604
    (qa, _ra) = numpy.linalg.qr(A)
    (qb, _rb) = numpy.linalg.qr(B)

    C = numpy.dot(numpy.transpose(qa), qb)

    (y, c, _zt) = numpy.linalg.svd(C)

    U = numpy.dot(qa, y)

    rank_c = len([s for s in c if numpy.abs(1.0 - s) < 1.e-10])

    return U[:, :rank_c]


class Cell(object):
    """Abstract class for a reference cell.  Provides accessors for
    geometry (vertex coordinates) as well as topology (orderings of
    vertices that make up edges, facecs, etc."""

    def __init__(self, shape, vertices, topology):
        """The constructor takes a shape code, the physical vertices expressed
        as a list of tuples of numbers, and the topology of a cell.

        The topology is stored as a dictionary of dictionaries t[i][j]
        where i is the dimension and j is the index of the facet of
        that dimension.  The result is a list of the vertices
        comprising the facet."""
        self.shape = shape
        self.vertices = vertices
        self.topology = topology

        # Given the topology, work out for each entity in the cell,
        # which other entities it contains.
        self.sub_entities = {}
        for dim, entities in topology.items():
            self.sub_entities[dim] = {}

            for e, v in entities.items():
                vertices = frozenset(v)
                sub_entities = []

                for dim_, entities_ in topology.items():
                    for e_, vertices_ in entities_.items():
                        if vertices.issuperset(vertices_):
                            sub_entities.append((dim_, e_))

                # Sort for the sake of determinism and by UFC conventions
                self.sub_entities[dim][e] = sorted(sub_entities)

        # Build connectivity dictionary for easier queries
        self.connectivity = {}
        for dim0, sub_entities in self.sub_entities.items():

            # Skip tensor product entities
            # TODO: Can we do something better?
            if isinstance(dim0, tuple):
                continue

            for entity, sub_sub_entities in sorted(sub_entities.items()):
                for dim1 in range(dim0+1):
                    d01_entities = filter(lambda x: x[0] == dim1, sub_sub_entities)
                    d01_entities = tuple(x[1] for x in d01_entities)
                    self.connectivity.setdefault((dim0, dim1), []).append(d01_entities)

    def _key(self):
        """Hashable object key data (excluding type)."""
        # Default: only type matters
        return None

    def __eq__(self, other):
        return type(self) == type(other) and self._key() == other._key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self), self._key()))

    def get_shape(self):
        """Returns the code for the element's shape."""
        return self.shape

    def get_vertices(self):
        """Returns an iterable of the element's vertices, each stored as a
        tuple."""
        return self.vertices

    def get_spatial_dimension(self):
        """Returns the spatial dimension in which the element lives."""
        return len(self.vertices[0])

    def get_topology(self):
        """Returns a dictionary encoding the topology of the element.

        The dictionary's keys are the spatial dimensions (0, 1, ...)
        and each value is a dictionary mapping."""
        return self.topology

    def get_connectivity(self):
        """Returns a dictionary encoding the connectivity of the element.

        The dictionary's keys are the spatial dimensions pairs ((1, 0),
        (2, 0), (2, 1), ...) and each value is a list with entities
        of second dimension ordered by local dim0-dim1 numbering."""
        return self.connectivity

    def get_vertices_of_subcomplex(self, t):
        """Returns the tuple of vertex coordinates associated with the labels
        contained in the iterable t."""
        return tuple([self.vertices[ti] for ti in t])

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  For tensor
        product cells, this a tuple of dimensions for each cell in the
        product.  For all other cells, this is the same as the spatial
        dimension."""
        raise NotImplementedError("Should be implemented in a subclass.")

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: `tuple` for tensor product cells, `int` otherwise
        """
        raise NotImplementedError("Should be implemented in a subclass.")

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: `tuple` for tensor product cells, `int` otherwise
        :arg entity_i: entity number (integer)
        """
        raise NotImplementedError("Should be implemented in a subclass.")


class Simplex(Cell):
    """Abstract class for a reference simplex."""

    def compute_normal(self, facet_i):
        """Returns the unit normal vector to facet i of codimension 1."""
        # Interval case
        if self.get_shape() == LINE:
            verts = numpy.asarray(self.vertices)
            v_i, = self.get_topology()[0][facet_i]
            n = verts[v_i] - verts[[1, 0][v_i]]
            return n / numpy.linalg.norm(n)

        # first, let's compute the span of the simplex
        # This is trivial if we have a d-simplex in R^d.
        # Not so otherwise.
        vert_vecs = [numpy.array(v)
                     for v in self.vertices]
        vert_vecs_foo = numpy.array([vert_vecs[i] - vert_vecs[0]
                                     for i in range(1, len(vert_vecs))])

        (u, s, vt) = numpy.linalg.svd(vert_vecs_foo)
        rank = len([si for si in s if si > 1.e-10])

        # this is the set of vectors that span the simplex
        spanu = u[:, :rank]

        t = self.get_topology()
        sd = self.get_spatial_dimension()
        vert_coords_of_facet = \
            self.get_vertices_of_subcomplex(t[sd-1][facet_i])

        # now I find everything normal to the facet.
        vcf = [numpy.array(foo)
               for foo in vert_coords_of_facet]
        facet_span = numpy.array([vcf[i] - vcf[0]
                                  for i in range(1, len(vcf))])
        (uf, sf, vft) = numpy.linalg.svd(facet_span)

        # now get the null space from vft
        rankfacet = len([si for si in sf if si > 1.e-10])
        facet_normal_space = numpy.transpose(vft[rankfacet:, :])

        # now, I have to compute the intersection of
        # facet_span with facet_normal_space
        foo = linalg_subspace_intersection(facet_normal_space, spanu)

        num_cols = foo.shape[1]

        if num_cols != 1:
            raise Exception("barf in normal computation")

        # now need to get the correct sign
        # get a vector in the direction
        nfoo = foo[:, 0]

        # what is the vertex not in the facet?
        verts_set = set(t[sd][0])
        verts_facet = set(t[sd - 1][facet_i])
        verts_diff = verts_set.difference(verts_facet)
        if len(verts_diff) != 1:
            raise Exception("barf in normal computation: getting sign")
        vert_off = verts_diff.pop()
        vert_on = verts_facet.pop()

        # get a vector from the off vertex to the facet
        v_to_facet = numpy.array(self.vertices[vert_on]) \
            - numpy.array(self.vertices[vert_off])

        if numpy.dot(v_to_facet, nfoo) > 0.0:
            return nfoo
        else:
            return -nfoo

    def compute_tangents(self, dim, i):
        """Computes tangents in any dimension based on differences
        between vertices and the first vertex of the i:th facet
        of dimension dim.  Returns a (possibly empty) list.
        These tangents are *NOT* normalized to have unit length."""
        t = self.get_topology()
        vs = list(map(numpy.array, self.get_vertices_of_subcomplex(t[dim][i])))
        ts = [v - vs[0] for v in vs[1:]]
        return ts

    def compute_normalized_tangents(self, dim, i):
        """Computes tangents in any dimension based on differences
        between vertices and the first vertex of the i:th facet
        of dimension dim.  Returns a (possibly empty) list.
        These tangents are normalized to have unit length."""
        ts = self.compute_tangents(dim, i)
        return [t / numpy.linalg.norm(t) for t in ts]

    def compute_edge_tangent(self, edge_i):
        """Computes the nonnormalized tangent to a 1-dimensional facet.
        returns a single vector."""
        t = self.get_topology()
        (v0, v1) = self.get_vertices_of_subcomplex(t[1][edge_i])
        return numpy.array(v1) - numpy.array(v0)

    def compute_normalized_edge_tangent(self, edge_i):
        """Computes the unit tangent vector to a 1-dimensional facet"""
        v = self.compute_edge_tangent(edge_i)
        return v / numpy.linalg.norm(v)

    def compute_face_tangents(self, face_i):
        """Computes the two tangents to a face.  Only implemented
        for a tetrahedron."""
        if self.get_spatial_dimension() != 3:
            raise Exception("can't get face tangents yet")
        t = self.get_topology()
        (v0, v1, v2) = list(map(numpy.array,
                                self.get_vertices_of_subcomplex(t[2][face_i])))
        return (v1 - v0, v2 - v0)

    def compute_face_edge_tangents(self, dim, entity_id):
        """Computes all the edge tangents of any k-face with k>=1.
        The result is a array of binom(dim+1,2) vectors.
        This agrees with `compute_edge_tangent` when dim=1.
        """
        vert_ids = self.get_topology()[dim][entity_id]
        vert_coords = [numpy.array(x)
                       for x in self.get_vertices_of_subcomplex(vert_ids)]
        edge_ts = []
        for source in range(dim):
            for dest in range(source + 1, dim + 1):
                edge_ts.append(vert_coords[dest] - vert_coords[source])
        return edge_ts

    def make_points(self, dim, entity_id, order):
        """Constructs a lattice of points on the entity_id:th
        facet of dimension dim.  Order indicates how many points to
        include in each direction."""
        if dim == 0:
            return (self.get_vertices()[entity_id], )
        elif 0 < dim < self.get_spatial_dimension():
            entity_verts = \
                self.get_vertices_of_subcomplex(
                    self.get_topology()[dim][entity_id])
            return make_lattice(entity_verts, order, 1)
        elif dim == self.get_spatial_dimension():
            return make_lattice(self.get_vertices(), order, 1)
        else:
            raise ValueError("illegal dimension")

    def volume(self):
        """Computes the volume of the simplex in the appropriate
        dimensional measure."""
        return volume(self.get_vertices())

    def volume_of_subcomplex(self, dim, facet_no):
        vids = self.topology[dim][facet_no]
        return volume(self.get_vertices_of_subcomplex(vids))

    def compute_scaled_normal(self, facet_i):
        """Returns the unit normal to facet_i of scaled by the
        volume of that facet."""
        dim = self.get_spatial_dimension()
        v = self.volume_of_subcomplex(dim - 1, facet_i)
        return self.compute_normal(facet_i) * v

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i."""
        assert facet_dim == self.get_spatial_dimension() - 1
        n = Simplex.compute_normal(self, facet_i)  # skip UFC overrides
        return n / numpy.linalg.norm(n, numpy.inf)

    def get_entity_transform(self, dim, entity):
        """Returns a mapping of point coordinates from the
        `entity`-th subentity of dimension `dim` to the cell.

        :arg dim: subentity dimension (integer)
        :arg entity: entity number (integer)
        """
        topology = self.get_topology()
        celldim = self.get_spatial_dimension()
        codim = celldim - dim
        if dim == 0:
            # Special case vertices.
            i, = topology[dim][entity]
            vertex = self.get_vertices()[i]
            return lambda point: vertex
        elif dim == celldim:
            assert entity == 0
            return lambda point: point

        try:
            subcell = self.construct_subelement(dim)
        except NotImplementedError:
            # Special case for 1D elements.
            x_c, = self.get_vertices_of_subcomplex(topology[0][entity])
            return lambda x: x_c

        subdim = subcell.get_spatial_dimension()

        assert subdim == celldim - codim

        # Entity vertices in entity space.
        v_e = numpy.asarray(subcell.get_vertices())

        A = numpy.zeros([subdim, subdim])

        for i in range(subdim):
            A[i, :] = (v_e[i + 1] - v_e[0])
            A[i, :] /= A[i, :].dot(A[i, :])

        # Entity vertices in cell space.
        v_c = numpy.asarray(self.get_vertices_of_subcomplex(topology[dim][entity]))

        B = numpy.zeros([celldim, subdim])

        for j in range(subdim):
            B[:, j] = (v_c[j + 1] - v_c[0])

        C = B.dot(A)

        offset = v_c[0] - C.dot(v_e[0])

        return lambda x: offset + C.dot(x)

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  Same as the
        spatial dimension."""
        return self.get_spatial_dimension()


# Backwards compatible name
ReferenceElement = Simplex


class UFCSimplex(Simplex):

    def get_facet_element(self):
        dimension = self.get_spatial_dimension()
        return self.construct_subelement(dimension - 1)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return ufc_simplex(dimension)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance)."""
        result = (sum(point) - epsilon <= 1)
        for c in point:
            result &= (c + epsilon >= 0)
        return result


class Point(Simplex):
    """This is the reference point."""

    def __init__(self):
        verts = ((),)
        topology = {0: {0: (0,)}}
        super(Point, self).__init__(POINT, verts, topology)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer). Must be zero.
        """
        assert dimension == 0
        return self


class DefaultLine(Simplex):
    """This is the reference line with vertices (-1.0,) and (1.0,)."""

    def __init__(self):
        verts = ((-1.0,), (1.0,))
        edges = {0: (0, 1)}
        topology = {0: {0: (0,), 1: (1,)},
                    1: edges}
        super(DefaultLine, self).__init__(LINE, verts, topology)

    def get_facet_element(self):
        raise NotImplementedError()


class UFCInterval(UFCSimplex):
    """This is the reference interval with vertices (0.0,) and (1.0,)."""

    def __init__(self):
        verts = ((0.0,), (1.0,))
        edges = {0: (0, 1)}
        topology = {0: {0: (0,), 1: (1,)},
                    1: edges}
        super(UFCInterval, self).__init__(LINE, verts, topology)


class DefaultTriangle(Simplex):
    """This is the reference triangle with vertices (-1.0,-1.0),
    (1.0,-1.0), and (-1.0,1.0)."""

    def __init__(self):
        verts = ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))
        edges = {0: (1, 2),
                 1: (2, 0),
                 2: (0, 1)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super(DefaultTriangle, self).__init__(TRIANGLE, verts, topology)

    def get_facet_element(self):
        return DefaultLine()


class UFCTriangle(UFCSimplex):
    """This is the reference triangle with vertices (0.0,0.0),
    (1.0,0.0), and (0.0,1.0)."""

    def __init__(self):
        verts = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        edges = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super(UFCTriangle, self).__init__(TRIANGLE, verts, topology)

    def compute_normal(self, i):
        "UFC consistent normal"
        t = self.compute_tangents(1, i)[0]
        n = numpy.array((t[1], -t[0]))
        return n / numpy.linalg.norm(n)


class IntrepidTriangle(Simplex):
    """This is the Intrepid triangle with vertices (0,0),(1,0),(0,1)"""

    def __init__(self):
        verts = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        edges = {0: (0, 1),
                 1: (1, 2),
                 2: (2, 0)}
        faces = {0: (0, 1, 2)}
        topology = {0: {0: (0,), 1: (1,), 2: (2,)},
                    1: edges, 2: faces}
        super(IntrepidTriangle, self).__init__(TRIANGLE, verts, topology)

    def get_facet_element(self):
        # I think the UFC interval is equivalent to what the
        # IntrepidInterval would be.
        return UFCInterval()


class DefaultTetrahedron(Simplex):
    """This is the reference tetrahedron with vertices (-1,-1,-1),
    (1,-1,-1),(-1,1,-1), and (-1,-1,1)."""

    def __init__(self):
        verts = ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
                 (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (1, 2),
                 1: (2, 0),
                 2: (0, 1),
                 3: (0, 3),
                 4: (1, 3),
                 5: (2, 3)}
        faces = {0: (1, 3, 2),
                 1: (2, 3, 0),
                 2: (3, 1, 0),
                 3: (0, 1, 2)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super(DefaultTetrahedron, self).__init__(TETRAHEDRON, verts, topology)

    def get_facet_element(self):
        return DefaultTriangle()


class IntrepidTetrahedron(Simplex):
    """This is the reference tetrahedron with vertices (0,0,0),
    (1,0,0),(0,1,0), and (0,0,1) used in the Intrepid project."""

    def __init__(self):
        verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (0, 1),
                 1: (1, 2),
                 2: (2, 0),
                 3: (0, 3),
                 4: (1, 3),
                 5: (2, 3)}
        faces = {0: (0, 1, 3),
                 1: (1, 2, 3),
                 2: (0, 3, 2),
                 3: (0, 2, 1)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super(IntrepidTetrahedron, self).__init__(TETRAHEDRON, verts, topology)

    def get_facet_element(self):
        return IntrepidTriangle()


class UFCTetrahedron(UFCSimplex):
    """This is the reference tetrahedron with vertices (0,0,0),
    (1,0,0),(0,1,0), and (0,0,1)."""

    def __init__(self):
        verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        vs = {0: (0, ),
              1: (1, ),
              2: (2, ),
              3: (3, )}
        edges = {0: (2, 3),
                 1: (1, 3),
                 2: (1, 2),
                 3: (0, 3),
                 4: (0, 2),
                 5: (0, 1)}
        faces = {0: (1, 2, 3),
                 1: (0, 2, 3),
                 2: (0, 1, 3),
                 3: (0, 1, 2)}
        tets = {0: (0, 1, 2, 3)}
        topology = {0: vs, 1: edges, 2: faces, 3: tets}
        super(UFCTetrahedron, self).__init__(TETRAHEDRON, verts, topology)

    def compute_normal(self, i):
        "UFC consistent normals."
        t = self.compute_tangents(2, i)
        n = numpy.cross(t[0], t[1])
        return -2.0 * n / numpy.linalg.norm(n)


class TensorProductCell(Cell):
    """A cell that is the product of FIAT cells."""

    def __init__(self, *cells):
        # Vertices
        vertices = tuple(tuple(chain(*coords))
                         for coords in product(*[cell.get_vertices()
                                                 for cell in cells]))

        # Topology
        shape = tuple(len(c.get_vertices()) for c in cells)
        topology = {}
        for dim in product(*[cell.get_topology().keys()
                             for cell in cells]):
            topology[dim] = {}
            topds = [cell.get_topology()[d]
                     for cell, d in zip(cells, dim)]
            for tuple_ei in product(*[sorted(topd)for topd in topds]):
                tuple_vs = list(product(*[topd[ei]
                                          for topd, ei in zip(topds, tuple_ei)]))
                vs = tuple(numpy.ravel_multi_index(numpy.transpose(tuple_vs), shape))
                topology[dim][tuple_ei] = vs
            # flatten entity numbers
            topology[dim] = dict(enumerate(topology[dim][key]
                                           for key in sorted(topology[dim])))

        super(TensorProductCell, self).__init__(TENSORPRODUCT, vertices, topology)
        self.cells = tuple(cells)

    def _key(self):
        return self.cells

    @staticmethod
    def _split_slices(lengths):
        n = len(lengths)
        delimiter = [0] * (n + 1)
        for i in range(n):
            delimiter[i + 1] = delimiter[i] + lengths[i]
        return [slice(delimiter[i], delimiter[i+1])
                for i in range(n)]

    def get_dimension(self):
        """Returns the subelement dimension of the cell, a tuple of
        dimensions for each cell in the product."""
        return tuple(c.get_dimension() for c in self.cells)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: dimension in each "direction" (tuple)
        """
        return TensorProductCell(*[c.construct_subelement(d)
                                   for c, d in zip(self.cells, dimension)])

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: subelement dimension (tuple)
        :arg entity_i: entity number (integer)
        """
        # unravel entity_i
        shape = tuple(len(c.get_topology()[d])
                      for c, d in zip(self.cells, dim))
        alpha = numpy.unravel_index(entity_i, shape)

        # entity transform on each subcell
        sct = [c.get_entity_transform(d, i)
               for c, d, i in zip(self.cells, dim, alpha)]

        slices = TensorProductCell._split_slices(dim)

        def transform(point):
            return list(chain(*[t(point[s])
                                for t, s in zip(sct, slices)]))
        return transform

    def volume(self):
        """Computes the volume in the appropriate dimensional measure."""
        return numpy.prod([c.volume() for c in self.cells])

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i of
        subelement dimension facet_dim."""
        assert len(facet_dim) == len(self.get_dimension())
        indicator = numpy.array(self.get_dimension()) - numpy.array(facet_dim)
        (cell_i,), = numpy.nonzero(indicator)

        n = []
        for i, c in enumerate(self.cells):
            if cell_i == i:
                n.extend(c.compute_reference_normal(facet_dim[i], facet_i))
            else:
                n.extend([0] * c.get_spatial_dimension())
        return numpy.asarray(n)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance)."""
        lengths = [c.get_spatial_dimension() for c in self.cells]
        assert len(point) == sum(lengths)
        slices = TensorProductCell._split_slices(lengths)
        return reduce(operator.and_,
                      (c.contains_point(point[s], epsilon=epsilon)
                       for c, s in zip(self.cells, slices)),
                      True)


class UFCQuadrilateral(Cell):
    """This is the reference quadrilateral with vertices
    (0.0, 0.0), (0.0, 1.0), (1.0, 0.0) and (1.0, 1.0)."""

    def __init__(self):
        product = TensorProductCell(UFCInterval(), UFCInterval())
        pt = product.get_topology()

        verts = product.get_vertices()
        topology = flatten_entities(pt)

        super(UFCQuadrilateral, self).__init__(QUADRILATERAL, verts, topology)

        self.product = product
        self.unflattening_map = compute_unflattening_map(pt)

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  Same as the
        spatial dimension."""
        return self.get_spatial_dimension()

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        if dimension == 2:
            return self
        elif dimension == 1:
            return UFCInterval()
        elif dimension == 0:
            return Point()
        else:
            raise ValueError("Invalid dimension: %d" % (dimension,))

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: entity dimension (integer)
        :arg entity_i: entity number (integer)
        """
        d, e = self.unflattening_map[(dim, entity_i)]
        return self.product.get_entity_transform(d, e)

    def volume(self):
        """Computes the volume in the appropriate dimensional measure."""
        return self.product.volume()

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i."""
        assert facet_dim == 1
        d, i = self.unflattening_map[(facet_dim, facet_i)]
        return self.product.compute_reference_normal(d, i)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance)."""
        return self.product.contains_point(point, epsilon=epsilon)


class UFCHexahedron(Cell):
    """This is the reference hexahedron with vertices
    (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
    (1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0) and (1.0, 1.0, 1.0)."""

    def __init__(self):
        product = TensorProductCell(UFCInterval(), UFCInterval(), UFCInterval())
        pt = product.get_topology()

        verts = product.get_vertices()
        topology = flatten_entities(pt)

        super(UFCHexahedron, self).__init__(HEXAHEDRON, verts, topology)

        self.product = product
        self.unflattening_map = compute_unflattening_map(pt)

    def get_dimension(self):
        """Returns the subelement dimension of the cell.  Same as the
        spatial dimension."""
        return self.get_spatial_dimension()

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        if dimension == 3:
            return self
        elif dimension == 2:
            return UFCQuadrilateral()
        elif dimension == 1:
            return UFCInterval()
        elif dimension == 0:
            return Point()
        else:
            raise ValueError("Invalid dimension: %d" % (dimension,))

    def get_entity_transform(self, dim, entity_i):
        """Returns a mapping of point coordinates from the
        `entity_i`-th subentity of dimension `dim` to the cell.

        :arg dim: entity dimension (integer)
        :arg entity_i: entity number (integer)
        """
        d, e = self.unflattening_map[(dim, entity_i)]
        return self.product.get_entity_transform(d, e)

    def volume(self):
        """Computes the volume in the appropriate dimensional measure."""
        return self.product.volume()

    def compute_reference_normal(self, facet_dim, facet_i):
        """Returns the unit normal in infinity norm to facet_i."""
        assert facet_dim == 2
        d, i = self.unflattening_map[(facet_dim, facet_i)]
        return self.product.compute_reference_normal(d, i)

    def contains_point(self, point, epsilon=0):
        """Checks if reference cell contains given point
        (with numerical tolerance)."""
        return self.product.contains_point(point, epsilon=epsilon)


def make_affine_mapping(xs, ys):
    """Constructs (A,b) such that x --> A * x + b is the affine
    mapping from the simplex defined by xs to the simplex defined by ys."""

    dim_x = len(xs[0])
    dim_y = len(ys[0])

    if len(xs) != len(ys):
        raise Exception("")

    # find A in R^{dim_y,dim_x}, b in R^{dim_y} such that
    # A xs[i] + b = ys[i] for all i

    mat = numpy.zeros((dim_x * dim_y + dim_y, dim_x * dim_y + dim_y), "d")
    rhs = numpy.zeros((dim_x * dim_y + dim_y,), "d")

    # loop over points
    for i in range(len(xs)):
        # loop over components of each A * point + b
        for j in range(dim_y):
            row_cur = i * dim_y + j
            col_start = dim_x * j
            col_finish = col_start + dim_x
            mat[row_cur, col_start:col_finish] = numpy.array(xs[i])
            rhs[row_cur] = ys[i][j]
            # need to get terms related to b
            mat[row_cur, dim_y * dim_x + j] = 1.0

    sol = numpy.linalg.solve(mat, rhs)

    A = numpy.reshape(sol[:dim_x * dim_y], (dim_y, dim_x))
    b = sol[dim_x * dim_y:]

    return A, b


def default_simplex(spatial_dim):
    """Factory function that maps spatial dimension to an instance of
    the default reference simplex of that dimension."""
    if spatial_dim == 1:
        return DefaultLine()
    elif spatial_dim == 2:
        return DefaultTriangle()
    elif spatial_dim == 3:
        return DefaultTetrahedron()
    else:
        raise RuntimeError("Can't create default simplex of dimension %s." % str(spatial_dim))


def ufc_simplex(spatial_dim):
    """Factory function that maps spatial dimension to an instance of
    the UFC reference simplex of that dimension."""
    if spatial_dim == 0:
        return Point()
    elif spatial_dim == 1:
        return UFCInterval()
    elif spatial_dim == 2:
        return UFCTriangle()
    elif spatial_dim == 3:
        return UFCTetrahedron()
    else:
        raise RuntimeError("Can't create UFC simplex of dimension %s." % str(spatial_dim))


def ufc_cell(cell):
    """Handle incoming calls from FFC."""

    # celltype could be a string or a cell.
    if isinstance(cell, str):
        celltype = cell
    else:
        celltype = cell.cellname()

    if " * " in celltype:
        # Tensor product cell
        return TensorProductCell(*map(ufc_cell, celltype.split(" * ")))
    elif celltype == "quadrilateral":
        return UFCQuadrilateral()
    elif celltype == "hexahedron":
        return UFCHexahedron()
    elif celltype == "vertex":
        return ufc_simplex(0)
    elif celltype == "interval":
        return ufc_simplex(1)
    elif celltype == "triangle":
        return ufc_simplex(2)
    elif celltype == "tetrahedron":
        return ufc_simplex(3)
    else:
        raise RuntimeError("Don't know how to create UFC cell of type %s" % str(celltype))


def volume(verts):
    """Constructs the volume of the simplex spanned by verts"""

    # use fact that volume of UFC reference element is 1/n!
    sd = len(verts) - 1
    ufcel = ufc_simplex(sd)
    ufcverts = ufcel.get_vertices()

    A, b = make_affine_mapping(ufcverts, verts)

    # can't just take determinant since, e.g. the face of
    # a tet being mapped to a 2d triangle doesn't have a
    # square matrix

    (u, s, vt) = numpy.linalg.svd(A)

    # this is the determinant of the "square part" of the matrix
    # (ie the part that maps the restriction of the higher-dimensional
    # stuff to UFC element
    p = numpy.prod([si for si in s if (si) > 1.e-10])

    return p / factorial(sd)


def tuple_sum(tree):
    """
    This function calculates the sum of elements in a tuple, it is needed to handle nested tuples in TensorProductCell.
    Example: tuple_sum(((1, 0), 1)) returns 2
    If input argument is not the tuple, returns input.
    """
    if isinstance(tree, tuple):
        return sum(map(tuple_sum, tree))
    else:
        return tree


def is_hypercube(cell):
    if isinstance(cell, (DefaultLine, UFCInterval, UFCQuadrilateral, UFCHexahedron)):
        return True
    elif isinstance(cell, TensorProductCell):
        return reduce(lambda a, b: a and b, [is_hypercube(c) for c in cell.cells])
    else:
        return False


def flatten_reference_cube(ref_el):
    """This function flattens a Tensor Product hypercube to the corresponding UFC hypercube"""
    flattened_cube = {2: UFCQuadrilateral(), 3: UFCHexahedron()}
    if numpy.sum(ref_el.get_dimension()) <= 1:
        # Just return point/interval cell arguments
        return ref_el
    else:
        # Handle cases where cell is a quad/cube constructed from a tensor product or
        # an already flattened element
        if is_hypercube(ref_el):
            return flattened_cube[numpy.sum(ref_el.get_dimension())]
        else:
            raise TypeError('Invalid cell type')


def flatten_entities(topology_dict):
    """This function flattens topology dict of TensorProductCell and entity_dofs dict of TensorProductElement"""

    flattened_entities = defaultdict(list)
    for dim in sorted(topology_dict.keys()):
        flat_dim = tuple_sum(dim)
        flattened_entities[flat_dim] += [v for k, v in sorted(topology_dict[dim].items())]

    return {dim: dict(enumerate(entities))
            for dim, entities in flattened_entities.items()}


def compute_unflattening_map(topology_dict):
    """This function returns unflattening map for the given tensor product topology dict."""

    counter = defaultdict(count)
    unflattening_map = {}

    for dim, entities in sorted(topology_dict.items()):
        flat_dim = tuple_sum(dim)
        for entity in entities:
            flat_entity = next(counter[flat_dim])
            unflattening_map[(flat_dim, flat_entity)] = (dim, entity)

    return unflattening_map
