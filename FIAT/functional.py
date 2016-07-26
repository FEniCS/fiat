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

# functionals require:
# - a degree of accuracy (-1 indicates that it works for all functions
#   such as point evaluation)
# - a reference element domain
# - type information

from __future__ import absolute_import, print_function, division

from collections import OrderedDict
from itertools import chain
import numpy
import sympy


def index_iterator(shp):
    """Constructs a generator iterating over all indices in
    shp in generalized column-major order  So if shp = (2,2), then we
    construct the sequence (0,0),(0,1),(1,0),(1,1)"""
    if len(shp) == 0:
        return
    elif len(shp) == 1:
        for i in range(shp[0]):
            yield [i]
    else:
        shp_foo = shp[1:]
        for i in range(shp[0]):
            for foo in index_iterator(shp_foo):
                yield [i] + foo

# also put in a "jet_dict" that maps
# pt --> {wt, multiindex, comp}
# the multiindex is an iterable of nonnegative
# integers


class Functional(object):
    """Class implementing an abstract functional.
    All functionals are discrete in the sense that
    the are written as a weighted sum of (components of) their
    argument evaluated at particular points."""

    def __init__(self, ref_el, target_shape, pt_dict, deriv_dict, functional_type):
        self.ref_el = ref_el
        self.target_shape = target_shape
        self.pt_dict = pt_dict
        self.deriv_dict = deriv_dict
        self.functional_type = functional_type
        if len(deriv_dict) > 0:
            per_point = list(chain(*deriv_dict.values()))
            alphas = [foo[1] for foo in per_point]
            self.max_deriv_order = max([sum(foo) for foo in alphas])
        else:
            self.max_deriv_order = 0

    def evaluate(self, f):
        """Obsolete and broken functional evaluation.

        To evaluate the functional, call it on the target function:

          functional(function)
        """
        raise AttributeError("To evaluate the functional just call it on a function.")

    def __call__(self, fn):
        raise NotImplementedError("Evaluation is not yet implemented for %s" % type(self))

    def get_point_dict(self):
        """Returns the functional information, which is a dictionary
        mapping each point in the support of the functional to a list
        of pairs containing the weight and component."""
        return self.pt_dict

    def get_reference_element(self):
        """Returns the reference element."""
        return self.ref_el

    def get_type_tag(self):
        """Returns the type of function (e.g. point evaluation or
        normal component, which is probably handy for clients of FIAT"""
        return self.functional_type

    # overload me in subclasses to make life easier!!
    def to_riesz(self, poly_set):
        """Constructs an array representation of the functional over
        the base of the given polynomial_set so that f(phi) for any
        phi in poly_set is given by a dot product."""
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        pt_dict = self.get_point_dict()

        pts = list(pt_dict.keys())

        # bfs is matrix that is pdim rows by num_pts cols
        # where pdim is the polynomial dimension

        bfs = es.tabulate(ed, pts)

        result = numpy.zeros(poly_set.coeffs.shape[1:], "d")

        # loop over points
        for j in range(len(pts)):
            pt_cur = pts[j]
            wc_list = pt_dict[pt_cur]

            # loop over expansion functions
            for i in range(bfs.shape[0]):
                for (w, c) in wc_list:
                    result[c][i] += w * bfs[i, j]

        if self.deriv_dict:
            raise NotImplementedError("Generic to_riesz implementation does not support derivatives")

        return result

    def tostr(self):
        return self.functional_type


class PointEvaluation(Functional):
    """Class representing point evaluation of scalar functions at a
    particular point x."""

    def __init__(self, ref_el, x):
        pt_dict = {x: [(1.0, tuple())]}
        Functional.__init__(self, ref_el, tuple(), pt_dict, {}, "PointEval")

    def __call__(self, fn):
        """Evaluate the functional on the function fn."""
        return fn(tuple(self.pt_dict.keys())[0])

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "u(%s)" % (','.join(x),)


class ComponentPointEvaluation(Functional):
    """Class representing point evaluation of a particular component
    of a vector function at a particular point x."""

    def __init__(self, ref_el, comp, shp, x):
        if len(shp) != 1:
            raise Exception("Illegal shape")
        if comp < 0 or comp >= shp[0]:
            raise Exception("Illegal component")
        self.comp = comp
        pt_dict = {x: [(1.0, (comp,))]}
        Functional.__init__(self, ref_el, shp, pt_dict, {},
                            "ComponentPointEval")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u[%d](%s)" % (self.comp, ','.join(x))


class PointDerivative(Functional):
    """Class representing point partial differentiation of scalar
    functions at a particular point x."""

    def __init__(self, ref_el, x, alpha):
        dpt_dict = {x: [(1.0, alpha, tuple())]}
        self.alpha = alpha
        self.order = sum(self.alpha)

        Functional.__init__(self, ref_el, tuple(), {}, dpt_dict, "PointDeriv")

    def __call__(self, fn):
        """Evaluate the functional on the function fn. Note that this depends
        on sympy being able to differentiate fn."""
        x = list(self.deriv_dict.keys())[0]

        X = sympy.DeferredVector('x')
        dX = numpy.asarray([X[i] for i in range(len(x))])

        dvars = tuple(d for d, a in zip(dX, self.alpha)
                      for count in range(a))

        return sympy.diff(fn(X), *dvars).evalf(subs=dict(zip(dX, x)))

    def to_riesz(self, poly_set):
        x = list(self.deriv_dict.keys())[0]

        X = sympy.DeferredVector('x')
        dx = numpy.asarray([X[i] for i in range(len(x))])

        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()

        bfs = es.tabulate(ed, [dx])[:, 0]

        # Expand the multi-index as a series of variables to
        # differentiate with respect to.
        dvars = tuple(d for d, a in zip(dx, self.alpha)
                      for count in range(a))

        return numpy.asarray([sympy.lambdify(X, sympy.diff(b, *dvars))(x)
                              for b in bfs])


class PointNormalDerivative(Functional):

    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        sd = ref_el.get_spatial_dimension()

        alphas = []
        for i in range(sd):
            alpha = [0] * sd
            alpha[i] = 1
            alphas.append(alpha)
        dpt_dict = {pt: [(n[i], alphas[i], tuple()) for i in range(sd)]}

        Functional.__init__(self, ref_el, tuple(), {}, dpt_dict, "PointNormalDeriv")


class IntegralMoment(Functional):
    """An IntegralMoment is a functional"""

    def __init__(self, ref_el, Q, f_at_qpts, comp=tuple(), shp=tuple()):
        """
        Create IntegralMoment

        *Arguments*

          ref_el
              The reference element (cell)
          Q (QuadratureRule)
              A quadrature rule for the integral
          f_at_qpts
              ???
          comp (tuple)
              A component ??? (Optional)
          shp  (tuple)
              The shape ??? (Optional)
        """
        qpts, qwts = Q.get_points(), Q.get_weights()
        pt_dict = OrderedDict()
        self.comp = comp
        for i in range(len(qpts)):
            pt_cur = tuple(qpts[i])
            pt_dict[pt_cur] = [(qwts[i] * f_at_qpts[i], comp)]
        Functional.__init__(self, ref_el, shp, pt_dict, {}, "IntegralMoment")

    def __call__(self, fn):
        """Evaluate the functional on the function fn."""
        pts = list(self.pt_dict.keys())
        wts = numpy.array([foo[0][0] for foo in list(self.pt_dict.values())])
        result = numpy.dot([fn(p) for p in pts], wts)

        if self.comp:
            result = result[self.comp]
        return result

    def to_riesz(self, poly_set):
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        pts = list(self.pt_dict.keys())
        bfs = es.tabulate(ed, pts)
        wts = numpy.array([foo[0][0] for foo in list(self.pt_dict.values())])
        result = numpy.zeros(poly_set.coeffs.shape[1:], "d")

        if len(self.comp) == 0:
            result[:] = numpy.dot(bfs, wts)
        else:
            result[self.comp, :] = numpy.dot(bfs, wts)

        return result


class FrobeniusIntegralMoment(Functional):

    def __init__(self, ref_el, Q, f_at_qpts):
        # f_at_qpts is num components x num_qpts
        if len(Q.get_points()) != f_at_qpts.shape[1]:
            raise Exception("Mismatch in number of quadrature points and values")

        # make sure that shp is same shape as f given
        shp = (f_at_qpts.shape[0],)

        qpts, qwts = Q.get_points(), Q.get_weights()
        pt_dict = {}
        for i in range(len(qpts)):
            pt_cur = tuple(qpts[i])
            pt_dict[pt_cur] = [(qwts[i] * f_at_qpts[j, i], (j,))
                               for j in range(f_at_qpts.shape[0])]

        Functional.__init__(self, ref_el, shp, pt_dict, {}, "FrobeniusIntegralMoment")


# point normals happen on a d-1 dimensional facet
# pt is the "physical" point on that facet
class PointNormalEvaluation(Functional):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1."""

    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        sd = ref_el.get_spatial_dimension()

        pt_dict = {pt: [(n[i], (i,)) for i in range(sd)]}

        shp = (sd,)
        Functional.__init__(self, ref_el, shp, pt_dict, {}, "PointNormalEval")


class PointEdgeTangentEvaluation(Functional):
    """Implements the evaluation of the tangential component of a
    vector at a point on a facet of dimension 1."""

    def __init__(self, ref_el, edge_no, pt):
        t = ref_el.compute_edge_tangent(edge_no)
        self.t = t
        sd = ref_el.get_spatial_dimension()
        pt_dict = {pt: [(t[i], (i,)) for i in range(sd)]}
        shp = (sd,)
        Functional.__init__(self, ref_el, shp, pt_dict, {}, "PointEdgeTangent")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.t)(%s)" % (','.join(x),)

    def to_riesz(self, poly_set):
        # should be singleton
        xs = list(self.pt_dict.keys())
        phis = poly_set.get_expansion_set().tabulate(poly_set.get_embedded_degree(), xs)
        return numpy.outer(self.t, phis)


class PointFaceTangentEvaluation(Functional):
    """Implements the evaluation of a tangential component of a
    vector at a point on a facet of codimension 1."""

    def __init__(self, ref_el, face_no, tno, pt):
        t = ref_el.compute_face_tangents(face_no)[tno]
        self.t = t
        self.tno = tno
        sd = ref_el.get_spatial_dimension()
        pt_dict = {pt: [(t[i], (i,)) for i in range(sd)]}
        shp = (sd,)
        Functional.__init__(self, ref_el, shp, pt_dict, {}, "PointFaceTangent")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.t%d)(%s)" % (self.tno, ','.join(x),)

    def to_riesz(self, poly_set):
        xs = list(self.pt_dict.keys())
        phis = poly_set.get_expansion_set().tabulate(poly_set.get_embedded_degree(), xs)
        return numpy.outer(self.t, phis)


class PointScaledNormalEvaluation(Functional):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1, where the normal is scaled by
    the volume of that facet."""

    def __init__(self, ref_el, facet_no, pt):
        self.n = ref_el.compute_scaled_normal(facet_no)
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)

        pt_dict = {pt: [(self.n[i], (i,)) for i in range(sd)]}
        Functional.__init__(self, ref_el, shp, pt_dict, {}, "PointScaledNormalEval")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.n)(%s)" % (','.join(x),)

    def to_riesz(self, poly_set):
        xs = list(self.pt_dict.keys())
        phis = poly_set.get_expansion_set().tabulate(poly_set.get_embedded_degree(), xs)
        return numpy.outer(self.n, phis)


class PointwiseInnerProductEvaluation(Functional):
    """
    This is a functional on symmetric 2-tensor fields. Let u be such a
    field, p be a point, and v,w be vectors. This implements the evaluation
    v^T u(p) w.

    Clearly v^iu_{ij}w^j = u_{ij}v^iw^j. Thus the value can be computed
    from the Frobenius inner product of u with wv^T. This gives the
    correct weights.
    """

    def __init__(self, ref_el, v, w, p):
        sd = ref_el.get_spatial_dimension()

        wvT = numpy.outer(w, v)

        pt_dict = {p: [(wvT[i][j], (i, j))
                       for i, j in index_iterator((sd, sd))]}

        shp = (sd, sd)
        Functional.__init__(self, ref_el, shp, pt_dict, {}, "PointwiseInnerProductEval")
