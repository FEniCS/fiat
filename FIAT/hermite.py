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

from . import finite_element, polynomial_set, dual_set , functional

class CubicHermiteDualSet( dual_set.DualSet ):
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

        # get jet at each vertex

        entity_ids[0] = {}
        for v in sorted( top[0] ):
            nodes.append( functional.PointEvaluation( ref_el , verts[v] ) )
            pd = functional.PointDerivative
            for i in range( sd ):
                alpha = [0] * sd
                alpha[i] = 1
                
                nodes.append( pd( ref_el , verts[v] , alpha ) )

            entity_ids[0][v] = list(range(cur,cur+1+sd))
            cur += sd + 1
                          
        # no edge dof
        entity_ids[1] = {}
        
        # face dof
        # point evaluation at barycenter
        entity_ids[2] = {}
        for f in sorted( top[2] ):
            pt = ref_el.make_points( 2 , f , 3 )[0]
            n = functional.PointEvaluation( ref_el , pt )
            nodes.append( n )
            entity_ids[2] = list(range(cur,cur+1))
            cur += 1
            

        for dim in range(3,sd+1):
            entity_ids[dim] = {}
            for facet in top[dim]:
                entity_ids[dim][facet] = []

        dual_set.DualSet.__init__( self , nodes , ref_el , entity_ids )

class CubicHermite( finite_element.FiniteElement ):
    """The Lagrange finite element.  It is what it is."""
    def __init__( self , ref_el ):
        poly_set = polynomial_set.ONPolynomialSet( ref_el , 3 )
        dual = CubicHermiteDualSet( ref_el  )
        finite_element.FiniteElement.__init__( self , poly_set , dual , 3 )

if __name__=="__main__":
    from . import reference_element
    T = reference_element.DefaultTetrahedron()
    U = CubicHermite( T )

    Ufs = U.get_nodal_basis()
    pts = T.make_lattice( 3 )
    print(pts)
    print(list(Ufs.tabulate(pts).values())[0])

