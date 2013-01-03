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

from . import finite_element, polynomial_set, dual_set , functional, P0

class DiscontinuousLagrangeDualSet( dual_set.DualSet ):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points.  This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""
    def __init__( self , ref_el , degree ):
        entity_ids = {}
        nodes = []

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()

        cur = 0
        for dim in sorted( top ):
            entity_ids[dim] = {}
            for entity in sorted( top[dim] ):
                pts_cur = ref_el.make_points( dim , entity , degree )
                nodes_cur = [ functional.PointEvaluation( ref_el , x ) \
                              for x in pts_cur ]
                nnodes_cur = len( nodes_cur )
                nodes +=  nodes_cur
                entity_ids[dim][entity]=[]
                cur += nnodes_cur

        entity_ids[dim][0] = list(range(len(nodes)))

        dual_set.DualSet.__init__( self , nodes , ref_el , entity_ids )

class HigherOrderDiscontinuousLagrange( finite_element.FiniteElement ):
    """The discontinuous Lagrange finite element.  It is what it is."""
    def __init__( self , ref_el , degree ):
        poly_set = polynomial_set.ONPolynomialSet( ref_el , degree )
        dual = DiscontinuousLagrangeDualSet( ref_el , degree )
        finite_element.FiniteElement.__init__( self , poly_set , dual , degree )

def DiscontinuousLagrange( ref_el , degree ):
    if degree == 0:
        return P0.P0( ref_el )
    else:
        return HigherOrderDiscontinuousLagrange( ref_el , degree )

if __name__=="__main__":
    from . import reference_element
    T = reference_element.DefaultTetrahedron()
    for k in range(2,3):
        U = DiscontinuousLagrange( T , k )

    Ufs = U.get_nodal_basis()
    pts = T.make_lattice( k )
    print(pts)
    for foo,bar in Ufs.tabulate( pts ,1 ).items():
        print(foo)
        print(bar)
        print()
