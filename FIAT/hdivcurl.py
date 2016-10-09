# Copyright (C) 2013 Andrew T. T. McRae (Imperial College London)
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
import types
from FIAT.tensor_product import TensorProductElement
from FIAT import functional


def Hdiv(element):
    if not isinstance(element, TensorProductElement):
        raise NotImplementedError

    if element.A.get_formdegree() is None or element.B.get_formdegree() is None:
        raise ValueError("form degree of sub-element was None (not set during initialisation), Hdiv cannot be done without this information")
    formdegree = element.A.get_formdegree() + element.B.get_formdegree()
    if formdegree != element.get_reference_element().get_spatial_dimension() - 1:
        raise ValueError("Tried to use Hdiv on a non-(n-1)-form element")

    newelement = TensorProductElement(element.A, element.B)  # make a copy to return

    # redefine value_shape()
    def value_shape(self):
        "Return the value shape of the finite element functions."
        return (self.get_reference_element().get_spatial_dimension(),)
    newelement.value_shape = types.MethodType(value_shape, newelement)

    # store old _mapping
    newelement._oldmapping = newelement._mapping

    # redefine _mapping
    newelement._mapping = "contravariant piola"

    # store formdegree
    newelement.formdegree = formdegree

    # redefine tabulate
    newelement.old_tabulate = newelement.tabulate

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""

        # don't duplicate what the old function does fine...
        old_result = self.old_tabulate(order, points, entity)
        new_result = {}
        sd = self.get_reference_element().get_spatial_dimension()
        for alpha in old_result.keys():
            temp_old = old_result[alpha]

            if self._oldmapping == "affine":
                temp = numpy.zeros((temp_old.shape[0], sd, temp_old.shape[1]), dtype=temp_old.dtype)
                # both constituents affine, i.e., they were 0 forms or n-forms.
                # to sum to n-1, we must have "0-form on an interval" crossed
                # with something discontinuous.
                # look for the (continuous) 0-form, and put the value there
                if self.A.get_formdegree() == 0:
                    # first element, so (-x, 0, ...)
                    # Sign flip to ensure that a positive value of the node
                    # means a vector field having a direction "to the left"
                    # relative to direction in which the nodes are placed on an
                    # edge in case of higher-order schemes.
                    # This is required for unstructured quadrilateral meshes.
                    temp[:, 0, :] = -temp_old[:, :]
                elif self.B.get_formdegree() == 0:
                    # second element, so (..., 0, x)
                    temp[:, -1, :] = temp_old[:, :]
                else:
                    raise Exception("Hdiv affine/affine form degrees broke")

            elif self._oldmapping == "contravariant piola":
                temp = numpy.zeros((temp_old.shape[0], sd, temp_old.shape[2]), dtype=temp_old.dtype)
                Asd = self.A.get_reference_element().get_spatial_dimension()
                # one component is affine, one is contravariant piola
                # the affine one must be an n-form, hence discontinuous
                # this component/these components get zeroed out
                if element.A.mapping()[0] == "contravariant piola":
                    # first element, so (x1, ..., xn, 0, ...)
                    temp[:, :Asd, :] = temp_old[:, :, :]
                elif element.B.mapping()[0] == "contravariant piola":
                    # second element, so (..., 0, x1, ..., xn)
                    temp[:, Asd:, :] = temp_old[:, :, :]
                else:
                    raise ValueError("Hdiv contravariant piola couldn't find an existing ConPi subelement")

            elif self._oldmapping == "covariant piola":
                temp = numpy.zeros((temp_old.shape[0], sd, temp_old.shape[2]), dtype=temp_old.dtype)
                # one component is affine, one is covariant piola
                # the affine one must be an n-form, hence discontinuous
                # this component/these components get zeroed out
                # the remaining part gets perped
                if element.A.mapping()[0] == "covariant piola":
                    Asd = self.A.get_reference_element().get_spatial_dimension()
                    if not Asd == 2:
                        raise ValueError("Must be 2d shape to automatically convert covariant to contravariant")
                    temp_perp = numpy.zeros(temp_old.shape, dtype=temp_old.dtype)
                    # first element, so (x2, -x1, 0, ...)
                    temp_perp[:, 0, :] = temp_old[:, 1, :]
                    temp_perp[:, 1, :] = -temp_old[:, 0, :]
                    temp[:, :Asd, :] = temp_perp[:, :, :]
                elif element.B.mapping()[0] == "covariant piola":
                    Bsd = self.B.get_reference_element().get_spatial_dimension()
                    if not Bsd == 2:
                        raise ValueError("Must be 2d shape to automatically convert covariant to contravariant")
                    temp_perp = numpy.zeros(temp_old.shape, dtype=temp_old.dtype)
                    # second element, so (..., 0, x2, -x1)
                    temp_perp[:, 0, :] = temp_old[:, 1, :]
                    temp_perp[:, 1, :] = -temp_old[:, 0, :]
                    temp[:, Asd:, :] = temp_old[:, :, :]
                else:
                    raise ValueError("Hdiv covariant piola couldn't find an existing CovPi subelement")
            new_result[alpha] = temp
        return new_result

    newelement.tabulate = types.MethodType(tabulate, newelement)

    # splat any PointEvaluation functionals.
    # they become a nasty mix of internal and external component DOFs
    if newelement._oldmapping == "affine":
        oldnodes = newelement.dual.nodes
        newnodes = []
        for node in oldnodes:
            if isinstance(node, functional.PointEvaluation):
                newnodes.append(functional.Functional(None, None, None, {}, "Undefined"))
            else:
                newnodes.append(node)
        newelement.dual.nodes = newnodes

    return newelement


def Hcurl(element):
    if not isinstance(element, TensorProductElement):
        raise NotImplementedError

    if element.A.get_formdegree() is None or element.B.get_formdegree() is None:
        raise ValueError("form degree of sub-element was None (not set during initialisation), Hcurl cannot be done without this information")
    formdegree = element.A.get_formdegree() + element.B.get_formdegree()
    if not (formdegree == 1):
        raise ValueError("Tried to use Hcurl on a non-1-form element")

    newelement = TensorProductElement(element.A, element.B)  # make a copy to return

    # redefine value_shape()
    def value_shape(self):
        "Return the value shape of the finite element functions."
        return (self.get_reference_element().get_spatial_dimension(),)
    newelement.value_shape = types.MethodType(value_shape, newelement)

    # store old _mapping
    newelement._oldmapping = newelement._mapping

    # redefine _mapping
    newelement._mapping = "covariant piola"

    # store formdegree
    newelement.formdegree = formdegree

    # redefine tabulate
    newelement.old_tabulate = newelement.tabulate

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points."""

        # don't duplicate what the old function does fine...
        old_result = self.old_tabulate(order, points, entity)
        new_result = {}
        sd = self.get_reference_element().get_spatial_dimension()
        for alpha in old_result.keys():
            temp_old = old_result[alpha]

            if self._oldmapping == "affine":
                temp = numpy.zeros((temp_old.shape[0], sd, temp_old.shape[1]), dtype=temp_old.dtype)
                # both constituents affine, i.e., they were 0 forms or n-forms.
                # to sum to 1, we must have "1-form on an interval" crossed with
                # a bunch of 0-forms (continuous).
                # look for the 1-form, and put the value in the other place
                if self.A.get_formdegree() == 1:
                    # first element, so (x, 0, ...)
                    # No sign flip here, nor at the other branch, to ensure that
                    # a positive value of the node means a vector field having
                    # the same direction as the direction in which the nodes are
                    # placed on an edge in case of higher-order schemes.
                    # This is required for unstructured quadrilateral meshes.
                    temp[:, 0, :] = temp_old[:, :]
                elif self.B.get_formdegree() == 1:
                    # second element, so (..., 0, x)
                    temp[:, -1, :] = temp_old[:, :]
                else:
                    raise Exception("Hcurl affine/affine form degrees broke")

            elif self._oldmapping == "covariant piola":
                temp = numpy.zeros((temp_old.shape[0], sd, temp_old.shape[2]), dtype=temp_old.dtype)
                Asd = self.A.get_reference_element().get_spatial_dimension()
                # one component is affine, one is covariant piola
                # the affine one must be an 0-form, hence continuous
                # this component/these components get zeroed out
                if element.A.mapping()[0] == "covariant piola":
                    # first element, so (x1, ..., xn, 0, ...)
                    temp[:, :Asd, :] = temp_old[:, :, :]
                elif element.B.mapping()[0] == "covariant piola":
                    # second element, so (..., 0, x1, ..., xn)
                    temp[:, Asd:, :] = temp_old[:, :, :]
                else:
                    raise ValueError("Hdiv contravariant piola couldn't find an existing ConPi subelement")

            elif self._oldmapping == "contravariant piola":
                temp = numpy.zeros((temp_old.shape[0], sd, temp_old.shape[2]), dtype=temp_old.dtype)
                # one component is affine, one is contravariant piola
                # the affine one must be an 0-form, hence continuous
                # this component/these components get zeroed out
                # the remaining part gets perped
                if element.A.mapping()[0] == "contravariant piola":
                    Asd = self.A.get_reference_element().get_spatial_dimension()
                    if not Asd == 2:
                        raise ValueError("Must be 2d shape to automatically convert contravariant to covariant")
                    temp_perp = numpy.zeros(temp_old.shape, dtype=temp_old.dtype)
                    # first element, so (-x2, x1, 0, ...)
                    temp_perp[:, 0, :] = -temp_old[:, 1, :]
                    temp_perp[:, 1, :] = temp_old[:, 0, :]
                    temp[:, :Asd, :] = temp_perp[:, :, :]
                elif element.B.mapping()[0] == "contravariant piola":
                    Bsd = self.B.get_reference_element().get_spatial_dimension()
                    if not Bsd == 2:
                        raise ValueError("Must be 2d shape to automatically convert contravariant to covariant")
                    temp_perp = numpy.zeros(temp_old.shape, dtype=temp_old.dtype)
                    # second element, so (..., 0, -x2, x1)
                    temp_perp[:, 0, :] = -temp_old[:, 1, :]
                    temp_perp[:, 1, :] = temp_old[:, 0, :]
                    temp[:, Asd:, :] = temp_old[:, :, :]
                else:
                    raise ValueError("Hcurl contravariant piola couldn't find an existing CovPi subelement")
            new_result[alpha] = temp
        return new_result

    newelement.tabulate = types.MethodType(tabulate, newelement)

    # splat any PointEvaluation functionals.
    # they become a nasty mix of internal and external component DOFs
    if newelement._oldmapping == "affine":
        oldnodes = newelement.dual.nodes
        newnodes = []
        for node in oldnodes:
            if isinstance(node, functional.PointEvaluation):
                newnodes.append(functional.Functional(None, None, None, {}, "Undefined"))
            else:
                newnodes.append(node)
        newelement.dual.nodes = newnodes

    return newelement
