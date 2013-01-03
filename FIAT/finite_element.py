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

import numpy
from .polynomial_set import PolynomialSet

class FiniteElement:
    """Class implementing Ciarlet's abstraction of a finite element
    being a domain, function space, and set of nodes."""
    def __init__( self , poly_set , dual , order, mapping="affine"):
        # first, compare ref_el of poly_set and dual
        # need to overload equality
        #if poly_set.get_reference_element() != dual.get_reference_element:
        #    raise Exception, ""

        # The order (degree) of the polynomial basis
        self.order = order

        self.ref_el = poly_set.get_reference_element()
        self.dual = dual

        # Appropriate mapping for the element space
        self._mapping = mapping

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

    def degree(self):
        "Return the degree of the (embedding) polynomial space."
        return self.poly_set.get_embedded_degree()

    def get_reference_element( self ):
        "Return the reference element for the finite element."
        return self.ref_el

    def get_nodal_basis( self ):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        return self.poly_set

    def get_dual_set( self ):
        "Return the dual for the finite element."
        return self.dual

    def get_order( self ):
        "Return the order of the element (may be different from the degree)"
        return self.order

    def dual_basis(self):
        """Return the dual basis (list of functionals) for the finite
        element."""
        return self.dual.get_nodes()

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self.dual.get_entity_ids()

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        return self.poly_set.get_coeffs()

    def mapping(self):
        """Return a list of appropriate mappings from the reference
        element to a physical element for each basis function of the
        finite element."""
        return [self._mapping]*self.space_dimension()

    def num_sub_elements(self):
        "Return the number of sub-elements."
        return 1

    def space_dimension(self):
        "Return the dimension of the finite element space."
        return self.poly_set.get_num_members()

    def tabulate(self, order, points):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""
        return self.poly_set.tabulate(points, order)

    def value_shape(self):
        "Return the value shape of the finite element functions."
        return self.poly_set.get_shape()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        return self.get_nodal_basis().get_dmats()

    def get_num_members(self, arg):
        "Return number of members of the expansion set."
        return self.get_nodal_basis().get_expansion_set().get_num_members(arg)
