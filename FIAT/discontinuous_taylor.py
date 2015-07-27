# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Colin Cotter (Imperial College London)
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

from FIAT import finite_element, polynomial_set, dual_set, functional, P0
from FIAT.reference_element import ufc_simplex
import numpy

class DiscontinuousTaylorDualSet( dual_set.DualSet ):
    """The dual basis for Taylor elements.  This class works for
    intervals.  Nodes are function and derivative evaluation
    at the midpoint. This is the discontinuous version where
    all nodes are topologically associated with the cell itself"""
    def __init__( self, ref_el, degree ):

        assert(ref_el.get_spatial_dimension()==1)

        entity_ids = {}
        nodes = []

        nodes.append( functional.PointEvaluation( ref_el, (0.,)))
        for k in range(1,degree+1):
            nodes.append( functional.PointDerivative( ref_el, (0.,), [k] ))
        
        entity_ids[0] = {}
        entity_ids[1] = {}
        entity_ids[0][0] = []
        entity_ids[0][1] = []
        entity_ids[1][0] = list(range(degree+1))

        dual_set.DualSet.__init__( self, nodes, ref_el, entity_ids )

class HigherOrderDiscontinuousTaylor( finite_element.FiniteElement ):
    """The discontinuous Taylor finite element. Use a Taylor basis for DG."""
    def __init__( self , ref_el , degree ):
        #this probably isn't used
        poly_set = polynomial_set.ONPolynomialSet( ref_el, degree )

        #set up dual space
        self.dual = DiscontinuousTaylorDualSet( ref_el, degree )

        #set up some other numbers
        self.formdegree = ref_el.get_spatial_dimension() # n-form
        self.polydegree = degree
        self.order = degree+1
        self.ref_el = ref_el
        self._mapping = "affine"

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self.polydegree

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented")
    
    def space_dimension(self):
        """Return the dimension of the finite element space."""
        # number of dofs just multiplies
        return self.formdegree

    def tabulate(self, order, points):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""

        points = numpy.array(points)
        result = {}
        
        #Tabulate the 0th derivatives
        vals0 = numpy.zeros((self.polydegree+1,len(points)))
        
        for p in range(self.polydegree+1):
            vals0[p,:] = (points-0.5)**p/numpy.math.factorial(p)

        result[0] = vals0

        vals = vals0.copy()
        #Tabulate the higher derivatives
        for k in range(1,order+1):
            vals[:] = 0.
            for p in range(self.polydegree+1):
                if(p-k>=0):
                    vals[p,:] = vals0[p-k,:]
                else:
                    vals[p,:] = 0.
            result[k] = vals

        return result

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return (0,)

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented")

#ufl/ufl/finiteelement/elementlist.py


def DiscontinuousTaylor( ref_el, degree ):
    if degree == 0:
        return P0.P0( ref_el )
    else:
        return HigherOrderDiscontinuousTaylor( ref_el, degree )

if __name__=="__main__":

    print("\n1D ----------------")
    T = ufc_simplex(1)
    element = DiscontinuousTaylor(T, 1)
    pts = [(0.0), (0.5), (1.0)]
    print("values = ", element.tabulate(0, pts))
    print("values = ", element.tabulate(1, pts))
