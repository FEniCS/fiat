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

import finite_element, polynomial_set, dual_set, functional, numpy

class ArgyrisDualSet( dual_set.DualSet ):
    def __init__( self , ref_el , degree ):
        entity_ids = {}
        nodes = []
        cur = 0

        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()

        if sd != 2:
            raise Exception("Illegal spatial dimension")

        pe = functional.PointEvaluation
        pd = functional.PointDerivative
        pnd = functional.PointNormalDerivative

        # get jet at each vertex

        entity_ids[0] = {}
        for v in sorted( top[0] ):
            nodes.append( pe( ref_el , verts[v] ) )

            # first derivatives
            for i in range( sd ):
                alpha = [0] * sd
                alpha[i] = 1
                nodes.append( pd( ref_el , verts[v] , alpha ) )

            # second derivatives
            alphas = [ [2,0] , [0,2] , [1,1] ]
            for alpha in alphas:
                nodes.append( pd( ref_el , verts[v] , alpha ) )


            entity_ids[0][v] = list(range(cur,cur+6))
            cur += 6

        # edge dof
        entity_ids[1] = {}
        for e in sorted( top[1] ):
            # normal derivatives at degree - 4 points on each edge
            ndpts = ref_el.make_points( 1 , e , degree - 3 )
            ndnds = [ pnd( ref_el , e , pt ) for pt in ndpts ]
            nodes.extend( ndnds )
            entity_ids[1][e] = list(range(cur,cur + len(ndpts)))
            cur += len( ndpts )

            # point value at degree-5 points on each edge
            if degree > 5:
                ptvalpts = ref_el.make_points( 1 , e , degree - 4 )
                ptvalnds = [ pe( ref_el , pt ) for pt in ptvalpts ]
                nodes.extend( ptvalnds )
                entity_ids[1][e] += list(range(cur,cur+len(ptvalpts)))
                cur += len( ptvalpts )

        # internal dof
        entity_ids[2] = {}
        if degree > 5:
            internalpts = ref_el.make_points( 2 , 0 , degree - 3 )
            internalnds = [ pe( ref_el , pt ) for pt in internalpts ]
            nodes.extend( internalnds )
            entity_ids[2][0] = list(range(cur,cur+len(internalpts)))
            cur += len(internalpts)

        dual_set.DualSet.__init__( self , nodes , ref_el , entity_ids )

class QuinticArgyrisDualSet( dual_set.DualSet ):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points."""
    def __init__( self , ref_el ):
        entity_ids = {}
        nodes = []
        cur = 0

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        if sd != 2:
            raise Exception("Illegal spatial dimension")

        pd = functional.PointDerivative

        # get jet at each vertex

        entity_ids[0] = {}
        for v in sorted( top[0] ):
            nodes.append( functional.PointEvaluation( ref_el , verts[v] ) )

            # first derivatives
            for i in range( sd ):
                alpha = [0] * sd
                alpha[i] = 1
                nodes.append( pd( ref_el , verts[v] , alpha ) )

            # second derivatives
            alphas = [ [2,0] , [0,2] , [1,1] ]
            for alpha in alphas:
                nodes.append( pd( ref_el , verts[v] , alpha ) )


            entity_ids[0][v] = list(range(cur,cur+6))
            cur += 6

        # edge dof -- normal at each edge midpoint
        entity_ids[1] = {}
        for e in sorted( top[1] ):
            pt = ref_el.make_points( 1 , e , 2 )[0]
            n = functional.PointNormalDerivative( ref_el , e , pt )
            nodes.append( n )
            entity_ids[1][e] = [cur]
            cur += 1



        dual_set.DualSet.__init__( self , nodes , ref_el , entity_ids )


class Argyris( finite_element.FiniteElement ):
    """The Argyris finite element."""
    def __init__( self , ref_el , degree ):
        poly_set = polynomial_set.ONPolynomialSet( ref_el , degree )
        dual = ArgyrisDualSet( ref_el , degree )
        finite_element.FiniteElement.__init__( self , poly_set , dual , degree )

class QuinticArgyris( finite_element.FiniteElement ):
    """The Argyris finite element."""
    def __init__( self , ref_el ):
        poly_set = polynomial_set.ONPolynomialSet( ref_el , 5 )
        dual = QuinticArgyrisDualSet( ref_el  )
        finite_element.FiniteElement.__init__( self , poly_set , dual , 5 )

if __name__=="__main__":
    from . import reference_element
    from . import lagrange
    T = reference_element.DefaultTriangle()
    for k in range(5,11):
        U = Argyris( T , k )
        U2 = lagrange.Lagrange( T , k )
        c = U.get_nodal_basis().get_coeffs()
        sigma = numpy.linalg.svd( c , compute_uv = 0)
        print("Argyris ",k, max(sigma) / min(sigma))
        c = U2.get_nodal_basis().get_coeffs()
        sigma = numpy.linalg.svd( c , compute_uv = 0)
        print("Lagrange ",k,max(sigma) / min(sigma ))
        print()
