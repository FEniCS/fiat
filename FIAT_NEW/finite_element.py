# Copyright 2008 by Robert C. Kirby (Texas Tech University)
# License: LGPL

import numpy
from polynomial_set import PolynomialSet

class FiniteElement:
    """Class implementing Ciarlet's abstraction of a finite element
    being a domain, function space, and set of nodes."""
    def __init__( self , poly_set , dual , order ):
        # first, compare ref_el of poly_set and dual
        # need to overload equality
        #if poly_set.get_reference_element() != dual.get_reference_element:
        #    raise Exception, ""

        # The order (degree) of the polynomial basis
        self.order = order

        self.ref_el = poly_set.get_reference_element()
        self.dual = dual

        # Appropriate mapping for the element space
        self._mapping = None

        # build generalized Vandermonde matrix
        old_coeffs = poly_set.get_coeffs()
        dualmat = dual.to_riesz( poly_set )

        shp = dualmat.shape
        if len( shp ) > 2:
            num_cols = numpy.prod( shp[1:] )

            A = numpy.reshape( dualmat , (dualmat.shape[0],num_cols) )
            B = numpy.reshape( old_coeffs , (old_coeffs.shape[0],num_cols ) )
        else:
            A = dualmat
            B = old_coeffs

        V = numpy.dot( A , numpy.transpose( B ) )
        self.V=V
        (u,s,vt) = numpy.linalg.svd( V )

        #print s
        #V = numpy.dot( dualmat , numpy.transpose( old_coeffs ) )

        Vinv = numpy.linalg.inv( V )

        new_coeffs_flat = numpy.dot( numpy.transpose( Vinv ) , B)

        new_shp = tuple( [ new_coeffs_flat.shape[0] ] \
                          + list( shp[1:] ) )
        new_coeffs = numpy.reshape( new_coeffs_flat , \
                                    new_shp )

        self.poly_set = PolynomialSet( self.ref_el , \
                                       poly_set.get_degree() , \
                                       poly_set.get_embedded_degree() , \
                                       poly_set.get_expansion_set() , \
                                       new_coeffs , \
                                       poly_set.get_dmats() )

        return



    def get_reference_element( self ):
        """Returns the reference element for the finite element."""
        return self.ref_el

    def get_nodal_basis( self ):
        """Returns the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        return self.poly_set

    def get_dual_set( self ):
        """Returns the dual for the finite element."""
        return self.dual

    def get_order( self ):
        return self.order

    def dual_basis(self):
        """Returns the dual basis (list of functionals) for the finite
        element."""
        return self.dual.get_nodes()

    def entity_dofs(self):
        return self.dual.get_entity_ids()

    def get_coeffs(self):
        return self.poly_set.get_coeffs()

    def mapping(self):
        """Returns the appropriate mapping from the reference element
        to a physical element for the finite element."""
        return self._mapping

    def num_sub_elements(self):
        return 1

    def space_dimension(self):
        return self.poly_set.get_num_members()

    def tabulate(self, order, points):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""
        return self.poly_set.tabulate(points, order)

    def value_shape(self):
        return self.poly_set.get_shape()
