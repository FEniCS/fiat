# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
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
#
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2014

import numpy
from .polynomial_set import PolynomialSet
from .quadrature import make_facet_quadrature

class FiniteElement:
    """Class implementing Ciarlet's abstraction of a finite element
    being a domain, function space, and set of nodes."""
    def __init__( self , poly_set , dual , order, formdegree=None, mapping="affine"):
        # first, compare ref_el of poly_set and dual
        # need to overload equality
        #if poly_set.get_reference_element() != dual.get_reference_element:
        #    raise Exception, ""

        # The order (degree) of the polynomial basis
        self.order = order
        self.formdegree = formdegree

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

            A = numpy.reshape( dualmat, (dualmat.shape[0], num_cols) )
            B = numpy.reshape( old_coeffs, (old_coeffs.shape[0], num_cols ) )
        else:
            A = dualmat
            B = old_coeffs

        V = numpy.dot( A, numpy.transpose( B ) )
        self.V=V
        (u, s, vt) = numpy.linalg.svd( V )

        Vinv = numpy.linalg.inv( V )

        new_coeffs_flat = numpy.dot( numpy.transpose( Vinv ), B)

        new_shp = tuple( [ new_coeffs_flat.shape[0] ] \
                          + list( shp[1:] ) )
        new_coeffs = numpy.reshape( new_coeffs_flat, \
                                    new_shp )

        self.poly_set = PolynomialSet( self.ref_el, \
                                       poly_set.get_degree(), \
                                       poly_set.get_embedded_degree(), \
                                       poly_set.get_expansion_set(), \
                                       new_coeffs, \
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

    def entity_closure_dofs(self):
        """Return the map of topological entities to degrees of
        freedom on the closure of those entities for the finite element."""
        return self.dual.get_entity_closure_ids()

    def facet_support_dofs(self):
        """Return the map of facet id to the degrees of freedom for which the
        corresponding basis functions take non-zero values."""
        if hasattr(self, "_facet_support_dofs"):
            return self._facet_support_dofs

        q = make_facet_quadrature(self.ref_el, max(2*self.degree(), 1))

        dim = self.ref_el.get_spatial_dimension()

        self._facet_support_dofs = {}

        for f in self.entity_dofs()[dim-1].keys():
            self._facet_support_dofs[f] = quadrature_support_dofs(self, q.get_points(f), q.get_weights())

        return self._facet_support_dofs

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        return self.poly_set.get_coeffs()

    def get_formdegree(self):
        """Return the degree of the associated form (FEEC)"""
        return self.formdegree

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


def quadrature_support_dofs(elem, points, weights):
    eps = 1.e-8  # Is this a safe value?

    dim = elem.ref_el.get_spatial_dimension()

    # Integrate the square of the basis functions on the facet.
    vals = numpy.double(elem.tabulate(0, points)[(0,) * dim])
    # Ints contains the square of the basis functions
    # integrated over the facet.
    if elem.value_shape():
        # Vector-valued functions.
        ints = numpy.dot(numpy.einsum("...ij,...ij->...j", vals, vals), weights)
    else:
        ints = numpy.dot(vals**2, weights)

    return [dof for dof, i in enumerate(ints) if i > eps]
