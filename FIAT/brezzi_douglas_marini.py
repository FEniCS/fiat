# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
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

from . import finite_element, raviart_thomas, quadrature, functional, \
    dual_set, reference_element, polynomial_set, nedelec

class BDMDualSet( dual_set.DualSet ):
    def __init__( self , ref_el , degree ):

        # Initialize containers for map: mesh_entity -> dof number and
        # dual basis
        entity_ids = {}
        nodes = []

        sd = ref_el.get_spatial_dimension()
        t = ref_el.get_topology()


        # Define each functional for the dual set
        # codimension 1 facets
        for i in range( len( t[sd-1] ) ):
            pts_cur = ref_el.make_points( sd - 1 , i , sd + degree )
            for j in range( len( pts_cur ) ):
                pt_cur = pts_cur[j]
                f = functional.PointScaledNormalEvaluation( ref_el , i , \
                                                            pt_cur )
                nodes.append( f )

        # internal nodes
        if degree > 1:
            Q = quadrature.make_quadrature( ref_el , 2 * (degree + 1) )
            qpts = Q.get_points()
            Nedel = nedelec.Nedelec( ref_el , degree - 1 )
            Nedfs = Nedel.get_nodal_basis()
            zero_index = tuple( [ 0 for i in range( sd ) ] )
            Ned_at_qpts = Nedfs.tabulate( qpts )[ zero_index ]

            for i in range( len( Ned_at_qpts ) ):
                phi_cur = Ned_at_qpts[i,:]
                l_cur = functional.FrobeniusIntegralMoment( ref_el , Q , \
                                                                phi_cur )
                nodes.append(l_cur)

        # sets vertices (and in 3d, edges) to have no nodes
        for i in range( sd - 1 ):
            entity_ids[i] = {}
            for j in range( len( t[i] ) ):
                entity_ids[i][j] = []

        cur = 0

        # set codimension 1 (edges 2d, faces 3d) dof
        pts_facet_0 = ref_el.make_points( sd - 1 , 0 , sd + degree )
        pts_per_facet = len( pts_facet_0 )

        entity_ids[sd-1] = {}
        for i in range( len( t[sd-1] ) ):
            entity_ids[sd-1][i] = list(range( cur , cur + pts_per_facet))
            cur += pts_per_facet

        # internal nodes, if applicable
        entity_ids[sd] = {0: []}

        if degree > 1:
            num_internal_nodes = len( Ned_at_qpts )
            entity_ids[sd][0] = list(range( cur , cur + num_internal_nodes))


        dual_set.DualSet.__init__( self , nodes , ref_el , entity_ids )

class BrezziDouglasMarini( finite_element.FiniteElement ):
    """The BDM element"""
    def __init__( self , ref_el , degree ):

        if degree < 1:
            raise Exception("BDM_k elements only valid for k >= 1")

        sd = ref_el.get_spatial_dimension()
        poly_set = polynomial_set.ONPolynomialSet( ref_el , degree , (sd,) )
        dual = BDMDualSet( ref_el , degree )
        finite_element.FiniteElement.__init__( self , poly_set , dual , degree,
                                               mapping="contravariant piola")

        return

if __name__=="__main__":
    T = reference_element.UFCTetrahedron()

    for k in range(1,3):
        print(k)
        BDM = BrezziDouglasMarini( T , k )
        print()
