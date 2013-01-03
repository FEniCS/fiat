from . import finite_element, quadrature, functional, \
    dual_set, reference_element, polynomial_set, lagrange

import numpy

class BDFMDualSet( dual_set.DualSet ):
    def __init__( self , ref_el , degree ):

        # Initialize containers for map: mesh_entity -> dof number and
        # dual basis
        entity_ids = {}
        nodes = []

        sd = ref_el.get_spatial_dimension()
        t = ref_el.get_topology()


        # Define each functional for the dual set
        # codimension 1 facet normals.
        # note this will die for degree greater than 1.
        for i in range( len( t[sd-1] ) ):
            pts_cur = ref_el.make_points( sd - 1 , i , sd + degree )
            for j in range( len( pts_cur ) ):
                pt_cur = pts_cur[j]
                f = functional.PointScaledNormalEvaluation( ref_el , i , \
                                                            pt_cur )
                nodes.append( f )

        # codimension 1 facet tangents.
        # because the tangent component is discontinuous, these actually
        # count as internal nodes.
        tangent_count=0
        for i in range( len( t[sd-1] ) ):
            pts_cur = ref_el.make_points( sd - 1 , i , sd + degree - 1 )
            tangent_count+=len( pts_cur )
            for j in range( len( pts_cur ) ):
                pt_cur = pts_cur[j]
                f = functional.PointEdgeTangentEvaluation( ref_el , i , \
                                                             pt_cur )
                nodes.append( f )

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

        # internal nodes
        entity_ids[sd] = {0: list(range(cur,cur+tangent_count))}
        cur+=tangent_count
    
        dual_set.DualSet.__init__( self , nodes , ref_el , entity_ids )

def BDFMSpace(ref_el, order):
    sd = ref_el.get_spatial_dimension()
    if sd !=2:
        raise Exception("BDFM_k elements only valid for dim 2")
    # Note that order will be 2.

    # Linear vector valued space. Since the embedding degree of this element
    # is 2, this is implemented by taking the quadratic space and selecting
    # the linear polynomials.
    vec_poly_set = polynomial_set.ONPolynomialSet( ref_el, order, (sd,) )
    # Linears are the first three polynomials in each dimension.
    vec_poly_set = vec_poly_set.take([0,1,2,6,7,8])

    # Scalar quadratic Lagrange element.
    lagrange_ele = lagrange.Lagrange(ref_el, order)
    # Select the dofs associated with the edges.
    edge_dofs_dict=lagrange_ele.dual.get_entity_ids()[sd-1]
    edge_dofs=numpy.array([(edge,dof) for edge,dofs in edge_dofs_dict.items()
                           for dof in dofs])

    tangent_polys=lagrange_ele.poly_set.take(edge_dofs[:,1])
    new_coeffs=numpy.zeros((tangent_polys.get_num_members(),sd,tangent_polys.coeffs.shape[-1]))
    
    # Outer product of the tangent vectors with the quadratic edge polynomials.
    for i,(edge, dof) in enumerate(edge_dofs):
        tangent=ref_el.compute_edge_tangent(edge)

        new_coeffs[i,:,:]=numpy.outer(tangent,tangent_polys.coeffs[i,:])
    
    bubble_set = polynomial_set.PolynomialSet( ref_el , \
                                               order, \
                                               order , \
                                               vec_poly_set.get_expansion_set() , \
                                               new_coeffs , \
                                               vec_poly_set.get_dmats() )    

    element_set =  polynomial_set.polynomial_set_union_normalized( bubble_set, vec_poly_set )
    return element_set
    

class BrezziDouglasFortinMarini( finite_element.FiniteElement ):
    """The BDFM element"""
    def __init__(self, ref_el, degree):

        if degree != 2:
            raise Exception("BDFM_k elements only valid for k == 2")

        poly_set = BDFMSpace(ref_el, degree)
        dual = BDFMDualSet(ref_el, degree-1)
        finite_element.FiniteElement.__init__(self, poly_set, dual, degree,
                                               mapping="contravariant piola")

        return

if __name__=="__main__":
    T = reference_element.UFCTriangle()

    BDFM = BrezziDouglasFortinMarini(T, 2)

    
