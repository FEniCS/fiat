# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Copyright (C) 2013 Andrew T. T. McRae
# Modified by Thomas H. Gibson, 2016
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

from __future__ import absolute_import, print_function, division

import numpy
from FIAT.finite_element import FiniteElement
from FIAT.reference_element import TensorProductCell
from FIAT.polynomial_set import mis
from FIAT import dual_set
from FIAT import functional


def _first_point(node):
    return tuple(node.get_point_dict().keys())[0]


def _first_point_pair(node):
    return tuple(node.get_point_dict().items())[0]


class TensorProductElement(FiniteElement):
    """Class implementing a finite element that is the tensor product
    of two existing finite elements."""

    def __init__(self, A, B):
        # set up simple things
        order = min(A.get_order(), B.get_order())
        if A.get_formdegree() is None or B.get_formdegree() is None:
            formdegree = None
        else:
            formdegree = A.get_formdegree() + B.get_formdegree()

        # set up reference element
        ref_el = TensorProductCell(A.get_reference_element(),
                                   B.get_reference_element())

        if A.mapping()[0] != "affine" and B.mapping()[0] == "affine":
            mapping = A.mapping()[0]
        elif B.mapping()[0] != "affine" and A.mapping()[0] == "affine":
            mapping = B.mapping()[0]
        elif A.mapping()[0] == "affine" and B.mapping()[0] == "affine":
            mapping = "affine"
        else:
            raise ValueError("check tensor product mappings - at least one must be affine")

        # set up entity_ids
        Adofs = A.entity_dofs()
        Bdofs = B.entity_dofs()
        Bsdim = B.space_dimension()
        entity_ids = {}

        for curAdim in Adofs:
            for curBdim in Bdofs:
                entity_ids[(curAdim, curBdim)] = {}
                dim_cur = 0
                for entityA in Adofs[curAdim]:
                    for entityB in Bdofs[curBdim]:
                        entity_ids[(curAdim, curBdim)][dim_cur] = \
                            [x*Bsdim + y for x in Adofs[curAdim][entityA]
                                for y in Bdofs[curBdim][entityB]]
                        dim_cur += 1

        # set up dual basis
        Anodes = A.dual_basis()
        Bnodes = B.dual_basis()

        # build the dual set by inspecting the current dual
        # sets item by item.
        # Currently supported cases:
        # PointEval x PointEval = PointEval [scalar x scalar = scalar]
        # PointScaledNormalEval x PointEval = PointScaledNormalEval [vector x scalar = vector]
        # ComponentPointEvaluation x PointEval [vector x scalar = vector]
        nodes = []
        for Anode in Anodes:
            if isinstance(Anode, functional.PointEvaluation):
                for Bnode in Bnodes:
                    if isinstance(Bnode, functional.PointEvaluation):
                        # case: PointEval x PointEval
                        # the PointEval functional just requires the
                        # coordinates. these are currently stored as
                        # the key of a one-item dictionary. we retrieve
                        # these by calling get_point_dict(), and
                        # use the concatenation to make a new PointEval
                        nodes.append(functional.PointEvaluation(ref_el, _first_point(Anode) + _first_point(Bnode)))
                    elif isinstance(Bnode, functional.IntegralMoment):
                        # dummy functional for product with integral moments
                        nodes.append(functional.Functional(None, None, None,
                                                           {}, "Undefined"))
                    elif isinstance(Bnode, functional.PointDerivative):
                        # dummy functional for product with point derivative
                        nodes.append(functional.Functional(None, None, None,
                                                           {}, "Undefined"))
                    else:
                        raise NotImplementedError("unsupported functional type")

            elif isinstance(Anode, functional.PointScaledNormalEvaluation):
                for Bnode in Bnodes:
                    if isinstance(Bnode, functional.PointEvaluation):
                        # case: PointScaledNormalEval x PointEval
                        # this could be wrong if the second shape
                        # has spatial dimension >1, since we are not
                        # explicitly scaling by facet size
                        if len(_first_point(Bnode)) > 1:
                            # TODO: support this case one day
                            raise NotImplementedError("PointScaledNormalEval x PointEval is not yet supported if the second shape has dimension > 1")
                        # We cannot make a new functional.PSNEval in
                        # the natural way, since it tries to compute
                        # the normal vector by itself.
                        # Instead, we create things manually, and
                        # call Functional() with these arguments
                        sd = ref_el.get_spatial_dimension()
                        # The pt_dict is a one-item dictionary containing
                        # the details of the functional.
                        # The key is the spatial coordinate, which
                        # is just a concatenation of the two parts.
                        # The value is a list of tuples, representing
                        # the normal vector (scaled by the volume of
                        # the facet) at that point.
                        # Each tuple looks like (foo, (i,)); the i'th
                        # component of the scaled normal is foo.

                        # The following line is only valid when the second
                        # shape has spatial dimension 1 (enforced above)
                        Apoint, Avalue = _first_point_pair(Anode)
                        pt_dict = {Apoint + _first_point(Bnode): Avalue + [(0.0, (len(Apoint),))]}

                        # The following line should be used in the
                        # general case
                        # pt_dict = {Anode.get_point_dict().keys()[0] + Bnode.get_point_dict().keys()[0]: Anode.get_point_dict().values()[0] + [(0.0, (ii,)) for ii in range(len(Anode.get_point_dict().keys()[0]), len(Anode.get_point_dict().keys()[0]) + len(Bnode.get_point_dict().keys()[0]))]}

                        # THE FOLLOWING IS PROBABLY CORRECT BUT UNTESTED
                        shp = (sd,)
                        nodes.append(functional.Functional(ref_el, shp, pt_dict, {}, "PointScaledNormalEval"))
                    else:
                        raise NotImplementedError("unsupported functional type")

            elif isinstance(Anode, functional.PointEdgeTangentEvaluation):
                for Bnode in Bnodes:
                    if isinstance(Bnode, functional.PointEvaluation):
                        # case: PointEdgeTangentEval x PointEval
                        # this is very similar to the case above, so comments omitted
                        if len(_first_point(Bnode)) > 1:
                            raise NotImplementedError("PointEdgeTangentEval x PointEval is not yet supported if the second shape has dimension > 1")
                        sd = ref_el.get_spatial_dimension()
                        Apoint, Avalue = _first_point_pair(Anode)
                        pt_dict = {Apoint + _first_point(Bnode): Avalue + [(0.0, (len(Apoint),))]}

                        # THE FOLLOWING IS PROBABLY CORRECT BUT UNTESTED
                        shp = (sd,)
                        nodes.append(functional.Functional(ref_el, shp, pt_dict, {}, "PointEdgeTangent"))
                    else:
                        raise NotImplementedError("unsupported functional type")

            elif isinstance(Anode, functional.ComponentPointEvaluation):
                for Bnode in Bnodes:
                    if isinstance(Bnode, functional.PointEvaluation):
                        # case: ComponentPointEval x PointEval
                        # the CptPointEval functional requires the component
                        # and the coordinates. very similar to PE x PE case.
                        sd = ref_el.get_spatial_dimension()
                        nodes.append(functional.ComponentPointEvaluation(ref_el, Anode.comp, (sd,), _first_point(Anode) + _first_point(Bnode)))
                    else:
                        raise NotImplementedError("unsupported functional type")

            elif isinstance(Anode, functional.FrobeniusIntegralMoment):
                for Bnode in Bnodes:
                    if isinstance(Bnode, functional.PointEvaluation):
                        # case: FroIntMom x PointEval
                        sd = ref_el.get_spatial_dimension()
                        pt_dict = {}
                        pt_old = Anode.get_point_dict()
                        for pt in pt_old:
                            pt_dict[pt+_first_point(Bnode)] = pt_old[pt] + [(0.0, sd-1)]
                        # THE FOLLOWING IS PROBABLY CORRECT BUT UNTESTED
                        shp = (sd,)
                        nodes.append(functional.Functional(ref_el, shp, pt_dict, {}, "FrobeniusIntegralMoment"))
                    else:
                        raise NotImplementedError("unsupported functional type")

            elif isinstance(Anode, functional.IntegralMoment):
                for Bnode in Bnodes:
                    if isinstance(Bnode, functional.PointEvaluation):
                        # case: IntMom x PointEval
                        sd = ref_el.get_spatial_dimension()
                        pt_dict = {}
                        pt_old = Anode.get_point_dict()
                        for pt in pt_old:
                            pt_dict[pt+_first_point(Bnode)] = pt_old[pt]
                        # THE FOLLOWING IS PROBABLY CORRECT BUT UNTESTED
                        shp = (sd,)
                        nodes.append(functional.Functional(ref_el, shp, pt_dict, {}, "IntegralMoment"))
                    else:
                        raise NotImplementedError("unsupported functional type")

            elif isinstance(Anode, functional.Functional):
                # this should catch everything else
                for Bnode in Bnodes:
                    nodes.append(functional.Functional(None, None, None, {}, "Undefined"))
            else:
                raise NotImplementedError("unsupported functional type")

        dual = dual_set.DualSet(nodes, ref_el, entity_ids)

        super(TensorProductElement, self).__init__(ref_el, dual, order, formdegree, mapping)
        # Set up constituent elements
        self.A = A
        self.B = B

        # degree for quadrature rule
        self.polydegree = max(A.degree(), B.degree())

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

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""
        if entity is None:
            entity = (self.ref_el.get_dimension(), 0)
        entity_dim, entity_id = entity

        shape = tuple(len(c.get_topology()[d])
                      for c, d in zip(self.ref_el.cells, entity_dim))
        idA, idB = numpy.unravel_index(entity_id, shape)

        # Factor the entity argument to get entities of the component elements
        entityA_dim, entityB_dim = entity_dim
        entityA = (entityA_dim, idA)
        entityB = (entityB_dim, idB)

        pointsAdim, pointsBdim = [c.get_spatial_dimension()
                                  for c in self.ref_el.construct_subelement(entity_dim).cells]
        pointsA = [point[:pointsAdim] for point in points]
        pointsB = [point[pointsAdim:pointsAdim + pointsBdim] for point in points]

        Asdim = self.A.ref_el.get_spatial_dimension()
        Bsdim = self.B.ref_el.get_spatial_dimension()
        # Note that for entities other than cells, the following
        # tabulations are already appropriately zero-padded so no
        # additional zero padding is required.
        Atab = self.A.tabulate(order, pointsA, entityA)
        Btab = self.B.tabulate(order, pointsB, entityB)
        npoints = len(points)

        # allow 2 scalar-valued FE spaces, or 1 scalar-valued,
        # 1 vector-valued. Combining 2 vector-valued spaces
        # into a tensor-valued space via an outer-product
        # seems to be a sensible general option, but I don't
        # know how to handle the nestedness of the arrays
        # if someone then tries to make a new "tensor finite
        # element" where one component is already a
        # tensor-valued space!
        A_valuedim = len(self.A.value_shape())  # scalar: 0, vector: 1
        B_valuedim = len(self.B.value_shape())  # scalar: 0, vector: 1
        if A_valuedim + B_valuedim > 1:
            raise NotImplementedError("tabulate does not support two vector-valued inputs")
        result = {}
        for i in range(order + 1):
            alphas = mis(Asdim+Bsdim, i)  # thanks, Rob!
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
                    temp = numpy.array([numpy.outer(
                                       Atab[alpha[0:Asdim]][..., j],
                                       Btab[alpha[Asdim:Asdim+Bsdim]][..., j])
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
                    temp = numpy.array([numpy.outer(
                                       Atab[alpha[0:Asdim]][..., j],
                                       Btab[alpha[Asdim:Asdim+Bsdim]][..., j])
                        for j in range(npoints)])
                    assert temp.shape[1] % 2 == 0
                    temp2 = temp.reshape((temp.shape[0],
                                          temp.shape[1]//2,
                                          2,
                                          temp.shape[2]))\
                        .transpose(0, 2, 1, 3)\
                        .reshape((temp.shape[0], 2, -1))\
                        .transpose(2, 1, 0)
                    result[alpha] = temp2
                elif A_valuedim == 0 and B_valuedim == 1:
                    # as above, with B's functions now vector-valued.
                    # we now do... [numpy.outer ... for ...] gives
                    # temp[point][f_i][g1x,g1y,g2x,g2y,...].
                    # reshape to temp[point][f_i][g_j][x/y]
                    # flatten middle: temp[point][full bf_i][x/y]
                    # transpose to temp[bf_i][x/y][point]
                    temp = numpy.array([numpy.outer(
                        Atab[alpha[0:Asdim]][..., j],
                        Btab[alpha[Asdim:Asdim+Bsdim]][..., j])
                        for j in range(len(Atab[alpha[0:Asdim]][0]))])
                    assert temp.shape[2] % 2 == 0
                    temp2 = temp.reshape((temp.shape[0], temp.shape[1],
                                          temp.shape[2]//2, 2))\
                        .reshape((temp.shape[0], -1, 2))\
                        .transpose(1, 2, 0)
                    result[alpha] = temp2
        return result

    def value_shape(self):
        """Return the value shape of the finite element functions."""
        if len(self.A.value_shape()) == 0 and len(self.B.value_shape()) == 0:
            return ()
        elif len(self.A.value_shape()) == 1 and len(self.B.value_shape()) == 0:
            return (self.A.value_shape()[0],)
        elif len(self.A.value_shape()) == 0 and len(self.B.value_shape()) == 1:
            return (self.B.value_shape()[0],)
        else:
            raise NotImplementedError("value_shape not implemented")

    def dmats(self):
        """Return dmats: expansion coefficients for basis function
        derivatives."""
        raise NotImplementedError("dmats not implemented")

    def get_num_members(self, arg):
        """Return number of members of the expansion set."""
        raise NotImplementedError("get_num_members not implemented")
