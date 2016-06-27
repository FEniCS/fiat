# Copyright (C) 2014 Andrew T. T. McRae
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

import numpy as np
from .finite_element import FiniteElement
from . import functional
from . import dual_set
from . import TensorProductElement


class HDivTrace(FiniteElement):
    """Class implementing the trace of hdiv elements -- a restriction
    which is non-zero only on cell facets.  The value taken is the
    dot product with the facet normal, in reference space."""

    def __init__(self, element):
        # check for hdiv element (contravariant piola mapping)
        if not element.mapping()[0] == "contravariant piola":
            raise ValueError("Can only take trace of Hdiv element")

        # TPEs not supported yet
        if isinstance(element, TensorProductElement):
            raise NotImplementedError("Trace of TFEs not supported yet")

        # Otherwise, let's go...
        self._element = element
        self.order = 0  # can't represent constant function
        self.ref_el = element.get_reference_element()
        sd = self.ref_el.get_spatial_dimension()
        self.formdegree = sd  # fully discontinuous
        self._mapping = "affine"

        # set up entity_ids
        new_ent_ids = {}
        self.dofmapping = []
        index = 0
        elt_dofs = element.entity_dofs()
        for ent_dim in elt_dofs:
            new_ent_ids[ent_dim] = {}
            for ent_dim_index in elt_dofs[ent_dim]:
                if ent_dim != sd - 1:
                    new_ent_ids[ent_dim][ent_dim_index] = []
                else:
                    # keep dofs, and add to dofmapping
                    new_ent_ids[ent_dim][ent_dim_index] = []
                    for foo in elt_dofs[ent_dim][ent_dim_index]:
                        self.dofmapping.append(foo)
                        new_ent_ids[ent_dim][ent_dim_index].append(index)
                        index += 1

        self.entity_ids = new_ent_ids
        self.fsdim = index

        # set up dual basis
        elt_nodes = element.dual_basis()
        nodes = []
        for i in self.dofmapping:
            if not isinstance(elt_nodes[i], functional.PointScaledNormalEvaluation) or isinstance(elt_nodes[i], functional.PointNormalEvaluation):
                raise RuntimeError("Not a PointNormalEvaluation dof, exiting")
            # create a PointEvaluation from each existing dof
            nodes.append(functional.PointEvaluation(self.ref_el, elt_nodes[i].get_point_dict().keys()[0]))
        self.dual = dual_set.DualSet(nodes, self.ref_el, self.entity_ids)

    def degree(self):
        """Return the degree of the (embedding) polynomial space."""
        return self._element.degree()

    def get_nodal_basis(self):
        """Return the nodal basis, encoded as a PolynomialSet object,
        for the finite element."""
        raise NotImplementedError("get_nodal_basis not implemented")

    def get_coeffs(self):
        """Return the expansion coefficients for the basis of the
        finite element."""
        raise NotImplementedError("get_coeffs not implemented")

    def num_sub_elements(self):
        """Return the number of sub-elements."""
        return 1

    def space_dimension(self):
        """Return the dimension of the finite element space."""
        return self.fsdim

    def tabulate(self, order, points):
        """Return tabulated values of basis functions at given points."""
        # if order > 0:
        #     raise ValueError("Can not tabulate derivatives")
        # for simplicity, only tabulate points on a single facet
        # (this can be changed later, but shouldn't be necessary,
        # since I think FFC asks for tabulation on one facet at a time)
        dim = len(points[0])
        tol = 1e-10
        # which facet are we on?
        # for i > 0, facet_i satisfies x[i-1] == 0.0
        # facet_0 satisfies x[0] + ... + x[n] = 1.0
        on_facet = [abs(points[0][foo]) < tol for foo in range(dim)]
        on_facet.insert(0, abs(sum(points[0]) - 1.0) < tol)
        # make sure the facet is unambiguous
        if on_facet.count(True) != 0:
            if on_facet.count(True) > 1:
                # This *has* to be caught, since we don't know which normal to
                # dot with
                raise RuntimeError("Attempted to tabulate a Trace space at an ambiguous location")

            # make sure all the points are on the same facet
            facetnum = on_facet.index(True)
            if facetnum > 0:
                check = [abs(points[i][facetnum - 1]) < tol for i in range(len(points))]
                if check.count(False) != 0:
                    raise RuntimeError("Attempted to tabulate a Trace space on multiple facets simultaneously")
            else:
                check = [abs(sum(points[i]) - 1.0) < tol for i in range(len(points))]
                if check.count(False) != 0:
                    raise RuntimeError("Attempted to tabulate a Trace space on multiple facets simultaneously")

            # finally, we are ready to go...
            elt_tab = self._element.tabulate(order, points)
            # elt_tab is a 3d array: [basis_fn][cpt][point]
            # we want to contract this to [basis_fn][point], by
            # dotting with the normal vector.  Also, we only
            # keep the basis functions according to the dofmapping list
            temp = np.zeros((self.fsdim, len(points)))
            # I don't know if this should be compute_normal or compute_scaled_normal
            # try this; if there are unexplained issues then try the other one.
            normal = self.get_reference_element().compute_scaled_normal(facetnum)
            for i in self.dofmapping:
                temp[i, :] = np.dot(normal, elt_tab[(0,)*dim][self.dofmapping[i]])
        else:
            # raise RuntimeError("Attempted to tabulate a Trace space away from a facet")
            temp = np.zeros((self.fsdim, len(points)))
            temp[:] = np.NAN

        # TODO: Many of these values should be 0.  Should we zero all
        # entries below a certain tolerance?
        if order == 0:
            return {(0,)*dim: temp}
        else:
            from .polynomial_set import mis
            tempdict = {(0,)*dim: temp}
            for i in range(order):
                alphas = mis(dim, i+1)
                for alpha in alphas:
                    tempdict[alpha] = np.zeros((self.fsdim, len(points)))
                    tempdict[alpha][:] = np.NAN
            return tempdict

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        return ()

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented")
