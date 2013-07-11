# Copyright (C) 2013 Robert C. Kirby's, Andrew T. T. McRae
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
from .finite_element import FiniteElement
from .reference_element import ReferenceElement, two_product_cell
from .polynomial_set import mis

class TensorFiniteElement( FiniteElement ):
    """Class implementing a finite element that is the tensor product
    of two existing finite elements."""
    def __init__( self , A , B ):
        self.A = A
        self.B = B

        # set up simple things
        self.polydegree = max(self.A.degree(), self.B.degree())
        self.order = min(self.A.get_order(), self.B.get_order())
        self.fsdim = self.A.space_dimension() * self.B.space_dimension()

        # set up reference element
        self.ref_el = two_product_cell(self.A.get_reference_element(), self.B.get_reference_element())

        if A.mapping()[0] <> "affine":
            self._mapping = A.mapping()[0]
        elif B.mapping()[0] <> "affine":
            self._mapping = B.mapping()[0]
        else:
            self._mapping = "affine"

        # set up entity_ids
        Adofs = self.A.entity_dofs()
        Bdofs = self.B.entity_dofs()
        Bsdim = self.B.space_dimension()
        self.entity_ids = {}

        for curAdim in Adofs:
            for curBdim in Bdofs:
                self.entity_ids[(curAdim,curBdim)] = {}
                dim_cur = 0
                for entityA in Adofs[curAdim]:
                    for entityB in Bdofs[curBdim]:
                        self.entity_ids[(curAdim,curBdim)][dim_cur] = \
                          [x*Bsdim + y for x in Adofs[curAdim][entityA] \
                          for y in Bdofs[curBdim][entityB]]
                        dim_cur += 1

        self.flattened_entity_ids = {}
        if (Bsdim == 1):
            vertlist = [0]
        else:
            vertlist = range(1, Bsdim+1)
            vertlist[0] = 0
            vertlist[-1] = 1

        for curAdim in Adofs:
            self.flattened_entity_ids[curAdim] = {}
            dim_cur = 0
            for entityA in Adofs[curAdim]:
                self.flattened_entity_ids[curAdim][dim_cur] = \
                  [x*Bsdim + y for y in vertlist \
                  for x in Adofs[curAdim][entityA]]
                dim_cur += 1

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polydegree

    def get_reference_element( self ):
        """Return the reference element for the finite element."""
        return self.ref_el

    def get_nodal_basis( self ):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented")

    def get_dual_set( self ):
        """Return the dual for the finite element."""
        raise NotImplementedError("get_dual_set not implemented")

    def get_order( self ):
        """Return the order of the element (may be different from the degree)."""
        return self.order

    def dual_basis(self):
        """Return the dual basis (list of functionals) for the finite
        element."""
        raise NotImplementedError("dual_basis not implemented")

    def entity_dofs(self):
        """Return the map of topological entities to degrees of
        freedom for the finite element."""
        return self.entity_ids

    def flattened_entity_dofs(self):
        """ Return the flattened (w.r.t. 2nd component) map of topological
        entities to degrees of freedom. Assumes product is something
        crossed with an interval"""
        return self.flattened_entity_ids

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented")

    def mapping(self):
        """Return a list of appropriate mappings from the reference
        element to a physical element for each basis function of the
        finite element."""
        return [self._mapping]*self.space_dimension()

    def num_sub_elements(self):
        """Return the number of sub-elements."""
        return 1

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        return self.fsdim

    def tabulate(self, order, points):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""

        Asdim = self.A.get_reference_element().get_spatial_dimension()
        Bsdim = self.B.get_reference_element().get_spatial_dimension()
        pointsA = [point[0:Asdim] for point in points]
        pointsB = [point[Asdim:Asdim+Bsdim] for point in points]
        Atab = self.A.tabulate(order, pointsA)
        Btab = self.B.tabulate(order, pointsB)
        npoints = len(points)

        # allow 2 scalar-valued FE spaces, or 1 scalar-valued,
        # 1 vector-valued. Combining 2 vector-valued spaces
        # into a tensor-valued space via an outer-product
        # seems to be a sensible general option, but I don't
        # know how to handle the nestedness of the arrays
        # if someone then tries to make a new "tensor finite
        # element" where one component is already a
        # tensor-valued space!
        A_valuedim = len(self.A.value_shape()) # scalar: 0, vector: 1
        B_valuedim = len(self.B.value_shape()) # scalar: 0, vector: 1
        if A_valuedim + B_valuedim > 1:
            raise Exception("tabulate does not support two vector-valued inputs... yet")
        result = {}
        for i in range( order + 1 ):
            alphas = mis( Asdim+Bsdim , i ) # thanks, Rob!
            for alpha in alphas:
                if A_valuedim == 0 and B_valuedim == 0:
                    # for each point, get outer product of (A's basis
                    # functions f1, f2, ... evaluated at that point)
                    # with (B's basis functions g1, g2, ... evaluated
                    # at that point). This gives temp[point][f_i][g_j].
                    # Flatten this, so bfs are
                    # in the order f1g1, f1g2, ..., f2g1, f2g2, ...
                    # which is compatible with the entity_dofs order.
                    # We now have temp[point][full basis function]
                    # Transpose this to get temp[bf][point],
                    # and we are done.
                    temp = numpy.array([numpy.outer( \
                        Atab[alpha[0:Asdim]][...,j], \
                        Btab[alpha[Asdim:Asdim+Bsdim]][...,j])\
                        .ravel() for j in range(npoints)])
                    result[alpha] = temp.transpose()
                elif A_valuedim == 1 and B_valuedim == 0:
                    # similar to above, except A's basis functions
                    # are now vector-valued. numpy.outer flattens the
                    # array, so it's like taking the OP of
                    # f1_x, f1_y, f2_x, f2_y, ... with g1, g2, ...
                    # this gives us
                    # temp[point][f1x, f1y, f2x, f2y, ...][g_j].
                    # reshape once to get temp[point][f_i][x/y][g_j]
                    # transpose to get temp[point][x/y][f_i][g_j]
                    # reshape to flatten the last two indices, this
                    # gives us temp[point][x/y][full bf_i]
                    # finally, transpose the first and last indices
                    # to get temp[bf_i][x/y][point], and we are done.
                    temp = numpy.array([numpy.outer( \
                    Atab[alpha[0:Asdim]][...,j], \
                    Btab[alpha[Asdim:Asdim+Bsdim]][...,j]) \
                    for j in range(npoints)])
                    result[alpha] = temp.reshape((temp.shape[0],temp.shape[1]/2,2,temp.shape[2]))\
                                    .transpose(0,2,1,3)\
                                    .reshape((temp.shape[0],2,-1))\
                                    .transpose(2,1,0)
                elif A_valuedim == 0 and B_valuedim == 1:
                    # as above, with B's functions now vector-valued.
                    # we now do... [numpy.outer ... for ...] gives
                    # temp[point][f_i][g1x,g1y,g2x,g2y,...].
                    # reshape to temp[point][f_i][g_j][x/y]
                    # flatten middle: temp[point][full bf_i][x/y]
                    # transpose to temp[bf_i][x/y][point]
                    temp = numpy.array([numpy.outer( \
                    Atab[alpha[0:Asdim]][...,j], \
                    Btab[alpha[Asdim:Asdim+Bsdim]][...,j]) \
                    for j in range(len(Atab[alpha[0:Asdim]][0]))])
                    result[alpha] = temp.reshape((temp.shape[0],temp.shape[1],temp.shape[2]/2,2))\
                                    .reshape((temp.shape[0],-1,2))\
                                    .transpose(1,2,0)
        return result

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        if len(self.A.value_shape()) == 0 and len(self.B.value_shape()) == 0:
            return ()
        elif len(self.A.value_shape()) == 1 and len(self.B.value_shape()) == 0:
            return (self.A.value_shape()[0]+self.B.get_reference_element            ().get_spatial_dimension(),)
        elif len(self.A.value_shape()) == 0 and len(self.B.value_shape()) == 1:
            return (self.B.value_shape()[0]+self.A.get_reference_element            ().get_spatial_dimension(),)
        else:
            raise NotImplementedError("value_shape not implemented")

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented")

if __name__=="__main__":
    from . import reference_element
    from . import lagrange
    from . import raviart_thomas
    S = reference_element.UFCTriangle()
    T = reference_element.UFCInterval()
    W = raviart_thomas.RaviartThomas(S, 1)
    X = lagrange.Lagrange(T, 3)
    Y = TensorFiniteElement(W, X)
    Z = TensorFiniteElement(Y, X)
