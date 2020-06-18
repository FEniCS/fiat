# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# functionals require:
# - a degree of accuracy (-1 indicates that it works for all functions
#   such as point evaluation)
# - a reference element domain
# - type information

from collections import OrderedDict
from itertools import chain
import numpy
import sympy

from FIAT.quadrature import GaussLegendreQuadratureLineRule, QuadratureRule


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

    def to_riesz(self, poly_set):
        x = list(self.deriv_dict.keys())[0]

        X = sympy.DeferredVector('x')
        dx = numpy.asarray([X[i] for i in range(len(x))])

        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()

        bfs = es.tabulate(ed, [dx])[:, 0]

        # We need the gradient dotted with the normal.
        return numpy.asarray(
            [sympy.lambdify(
                X, sum([sympy.diff(b, dxi)*ni
                        for dxi, ni in zip(dx, self.n)]))(x)
             for b in bfs])


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
        self.Q = Q
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


class IntegralMomentOfNormalDerivative(Functional):
    """Functional giving normal derivative integrated against some function on a facet."""

    def __init__(self, ref_el, facet_no, Q, f_at_qpts):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        self.f_at_qpts = f_at_qpts
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        # map points onto facet

        fmap = ref_el.get_entity_transform(sd-1, facet_no)
        qpts, qwts = Q.get_points(), Q.get_weights()
        dpts = [fmap(pt) for pt in qpts]
        self.dpts = dpts

        dpt_dict = OrderedDict()

        alphas = [[1 if j == i else 0 for j in range(sd)] for i in range(sd)]
        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*n[i], alphas[i], tuple()) for i in range(sd)]

        Functional.__init__(self, ref_el, tuple(),
                            {}, dpt_dict, "IntegralMomentOfNormalDerivative")

    def to_riesz(self, poly_set):
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()

        result = numpy.zeros(es.get_num_members(ed))
        sd = self.ref_el.get_spatial_dimension()

        X = sympy.DeferredVector('x')
        dX = numpy.asarray([X[i] for i in range(sd)])

        # evaluate bfs symbolically
        bfs = es.tabulate(ed, [dX])[:, 0]

        n = self.n
        qwts = self.Q.get_weights()

        for i in range(len(result)):
            thing = sympy.lambdify(
                X, sum([sympy.diff(bfs[i], dxi)*ni
                        for dxi, ni in zip(dX, n)]))

            for j, pt in enumerate(self.deriv_dict.keys()):
                result[i] += qwts[j] * self.f_at_qpts[j] * thing(pt)

        return result


class IntegralLegendreDirectionalMoment(Functional):
    """Momement of v.s against a Legendre polynomial over an edge"""
    def __init__(self, cell, s, entity, mom_deg, comp_deg, nm=""):
        from FIAT.quadrature import GaussLegendreQuadratureLineRule, QuadratureRule
        from FIAT.reference_element import UFCInterval as interval
        sd = cell.get_spatial_dimension()
        assert sd == 2
        shp = (sd,)
        quadpoints = comp_deg + 1
        Q = GaussLegendreQuadratureLineRule(interval(), quadpoints)
        legendre = numpy.polynomial.legendre.legval(2*Q.get_points()-1, [0]*mom_deg + [1])
        f_at_qpts = numpy.array([s*legendre[i] for i in range(quadpoints)])
        fmap = cell.get_entity_transform(sd-1, entity)
        mappedqpts = [fmap(pt) for pt in Q.get_points()]
        mappedQ = QuadratureRule(cell, mappedqpts, Q.get_weights())
        qwts = mappedQ.wts
        qpts = mappedQ.pts

        pt_dict = OrderedDict()

        for k in range(len(qpts)):
            pt_cur = tuple(qpts[k])
            pt_dict[pt_cur] = [(qwts[k] * f_at_qpts[k, i], (i,))
                               for i in range(2)]

        Functional.__init__(self, cell, shp, pt_dict, {}, nm)


class IntegralLegendreNormalMoment(IntegralLegendreDirectionalMoment):
    """Momement of v.n against a Legendre polynomial over an edge"""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        # n = cell.compute_normal(entity)
        n = cell.compute_scaled_normal(entity)
        IntegralLegendreDirectionalMoment.__init__(self, cell, n, entity,
                                                   mom_deg, comp_deg,
                                                   "IntegralLegendreNormalMoment")


class IntegralLegendreTangentialMoment(IntegralLegendreDirectionalMoment):
    """Momement of v.t against a Legendre polynomial over an edge"""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        # t = cell.compute_normalized_edge_tangent(entity)
        t = cell.compute_edge_tangent(entity)
        IntegralLegendreDirectionalMoment.__init__(self, cell, t, entity,
                                                   mom_deg, comp_deg,
                                                   "IntegralLegendreTangentialMoment")


class IntegralLegendreBidirectionalMoment(Functional):
    """Moment of dot(s1, dot(tau, s2)) against Legendre on entity, multiplied by the size of the reference facet"""
    def __init__(self, cell, s1, s2, entity, mom_deg, comp_deg, nm=""):
        from FIAT.reference_element import UFCInterval as interval
        # mom_deg is degree of moment, comp_deg is the total degree of
        # polynomial you might need to integrate (or something like that)
        sd = cell.get_spatial_dimension()
        shp = (sd, sd)

        s1s2T = numpy.outer(s1, s2)
        quadpoints = comp_deg + 1
        Q = GaussLegendreQuadratureLineRule(interval(), quadpoints)

        # The volume squared gets the Jacobian mapping from line interval
        # and the edge length into the functional.
        legendre = numpy.polynomial.legendre.legval(2*Q.get_points()-1, [0]*mom_deg + [1]) * numpy.abs(cell.volume_of_subcomplex(1, entity))**2

        f_at_qpts = numpy.array([s1s2T*legendre[i] for i in range(quadpoints)])

        # Map the quadrature points
        fmap = cell.get_entity_transform(sd-1, entity)
        mappedqpts = [fmap(pt) for pt in Q.get_points()]
        mappedQ = QuadratureRule(cell, mappedqpts, Q.get_weights())

        pt_dict = OrderedDict()

        qpts = mappedQ.pts
        qwts = mappedQ.wts

        for k in range(len(qpts)):
            pt_cur = tuple(qpts[k])
            pt_dict[pt_cur] = [(qwts[k] * f_at_qpts[k, i, j], (i, j))
                               for (i, j) in index_iterator(shp)]

        Functional.__init__(self, cell, shp, pt_dict, {}, nm)


class IntegralLegendreNormalNormalMoment(IntegralLegendreBidirectionalMoment):
    """Moment of dot(n, dot(tau, n)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_normal(entity)
        IntegralLegendreBidirectionalMoment.__init__(self, cell, n, n,
                                                     entity, mom_deg, comp_deg,
                                                     "IntegralNormalNormalLegendreMoment")


class IntegralLegendreNormalTangentialMoment(IntegralLegendreBidirectionalMoment):
    """Moment of dot(n, dot(tau, n)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_normal(entity)
        t = cell.compute_normalized_edge_tangent(entity)
        IntegralLegendreBidirectionalMoment.__init__(self, cell, n, t,
                                                     entity, mom_deg, comp_deg,
                                                     "IntegralNormalTangentialLegendreMoment")


class IntegralMomentOfDivergence(Functional):
    def __init__(self, ref_el, Q, f_at_qpts):
        self.f_at_qpts = f_at_qpts
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        qpts, qwts = Q.get_points(), Q.get_weights()
        dpts = qpts
        self.dpts = dpts

        dpt_dict = OrderedDict()

        alphas = [[1 if j == i else 0 for j in range(sd)] for i in range(sd)]
        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*f_at_qpts[j], alphas[i], (i,)) for i in range(sd)]

        Functional.__init__(self, ref_el, tuple(),
                            {}, dpt_dict, "IntegralMomentOfDivergence")

    def to_riesz(self, poly_set):
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()

        sd = self.ref_el.get_spatial_dimension()
        result = numpy.zeros(poly_set.coeffs.shape[1:], "d")
        X = sympy.DeferredVector('x')
        dX = numpy.asarray([X[i] for i in range(sd)])

        # evaluate bfs symbolically
        bfs = es.tabulate(ed, [dX])[:, 0]
        qwts = self.Q.get_weights()

        for j in range(len(bfs)):
            grad_phi = [sympy.lambdify(X, sympy.diff(bfs[j], dXcur))
                        for dXcur in dX]
            for i in range(sd):
                for k, pt in enumerate(self.deriv_dict.keys()):
                    result[i, j] += qwts[k] * self.f_at_qpts[k] * grad_phi[i](pt)

        return result


class IntegralMomentOfTensorDivergence(Functional):
    """Like IntegralMomentOfDivergence, but on symmetric tensors."""

    def __init__(self, ref_el, Q, f_at_qpts):
        self.f_at_qpts = f_at_qpts
        self.Q = Q
        qpts, qwts = Q.get_points(), Q.get_weights()
        nqp = len(qpts)
        dpts = qpts
        self.dpts = dpts

        assert len(f_at_qpts.shape) == 2
        assert f_at_qpts.shape[0] == 2
        assert f_at_qpts.shape[1] == nqp

        sd = ref_el.get_spatial_dimension()

        dpt_dict = OrderedDict()

        alphas = [[1 if j == i else 0 for j in range(sd)] for i in range(sd)]
        for q, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[q]*f_at_qpts[i, q], alphas[j], (i, j)) for i in range(2) for j in range(2)]

        Functional.__init__(self, ref_el, tuple(),
                            {}, dpt_dict, "IntegralMomentOfDivergence")

    def to_riesz(self, poly_set):
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()

        sd = self.ref_el.get_spatial_dimension()
        result = numpy.zeros(poly_set.coeffs.shape[1:], "d")
        X = sympy.DeferredVector('x')
        dX = numpy.asarray([X[i] for i in range(sd)])

        # evaluate bfs symbolically
        bfs = es.tabulate(ed, [dX])[:, 0]
        qwts = self.Q.get_weights()

        for k in range(len(bfs)):
            grad_phi = [sympy.lambdify(X, sympy.diff(bfs[k], dXcur))
                        for dXcur in dX]
            for i in range(sd):
                for j in range(sd):
                    for q, pt in enumerate(self.deriv_dict.keys()):
                        result[i, j, k] += qwts[q] * self.f_at_qpts[i, q] * grad_phi[j](pt)

        return result


class FrobeniusIntegralMoment(Functional):

    def __init__(self, ref_el, Q, f_at_qpts):
        # f_at_qpts is (some shape) x num_qpts
        shp = tuple(f_at_qpts.shape[:-1])
        if len(Q.get_points()) != f_at_qpts.shape[-1]:
            raise Exception("Mismatch in number of quadrature points and values")

        qpts, qwts = Q.get_points(), Q.get_weights()
        pt_dict = {}

        for i, (pt_cur, wt_cur) in enumerate(zip(map(tuple, qpts), qwts)):
            pt_dict[pt_cur] = []
            for alfa in index_iterator(shp):
                qpidx = tuple(alfa + [i])
                pt_dict[pt_cur].append((wt_cur * f_at_qpts[qpidx], tuple(alfa)))

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


class TensorBidirectionalMomentInnerProductEvaluation(Functional):
    r"""
    This is a functional on symmetric 2-tensor fields. Let u be such a
    field, f a function tabulated at points, and v,w be vectors. This implements the evaluation
    \int v^T u(x) w f(x).

    Clearly v^iu_{ij}w^j = u_{ij}v^iw^j. Thus the value can be computed
    from the Frobenius inner product of u with wv^T. This gives the
    correct weights.
    """

    def __init__(self, ref_el, v, w, Q, f_at_qpts, comp_deg):
        sd = ref_el.get_spatial_dimension()

        wvT = numpy.outer(w, v)

        qpts, qwts = Q.get_points(), Q.get_weights()

        pt_dict = {}
        for k, pt in enumerate(map(tuple(qpts))):
            pt_dict[pt] = []
            for i, j in index_iterator((sd, sd)):
                pt_dict[pt].append((qwts[k] * wvT[i][j] * f_at_qpts[i, j, k]),
                                   (i, j))

        shp = (sd, sd)
        Functional.__init__(self, ref_el, shp, pt_dict, {}, "TensorBidirectionalMomentInnerProductEvaluation")


class IntegralMomentOfNormalEvaluation(Functional):

    r"""
    \int_F v\cdot n p ds
    p \in Polynomials
    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        # scaling on the normal is ok because edge length then weights
        # the reference element quadrature appropriately
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        transform = ref_el.get_entity_transform(sd - 1, facet)
        pts = tuple(map(lambda p: tuple(transform(p)), Q.get_points()))
        weights = Q.get_weights()
        pt_dict = OrderedDict()
        for pt, wgt, phi in zip(pts, weights, P_at_qpts):
            pt_dict[pt] = [(wgt*phi*n[i], (i, )) for i in range(sd)]
        super().__init__(ref_el, (sd, ), pt_dict, {}, "IntegralMomentOfScaledNormalEvaluation")


class IntegralMomentOfTangentialEvaluation(Functional):

    r"""
    \int_F v\cdot n p ds
    p \in Polynomials
    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        # scaling on the tangent is ok because edge length then weights
        # the reference element quadrature appropriately
        sd = ref_el.get_spatial_dimension()
        assert sd == 2
        t = ref_el.compute_edge_tangent(facet)
        transform = ref_el.get_entity_transform(sd - 1, facet)
        pts = tuple(map(lambda p: tuple(transform(p)), Q.get_points()))
        weights = Q.get_weights()
        pt_dict = OrderedDict()
        for pt, wgt, phi in zip(pts, weights, P_at_qpts):
            pt_dict[pt] = [(wgt*phi*t[i], (i, )) for i in range(sd)]
        super().__init__(ref_el, (sd, ), pt_dict, {}, "IntegralMomentOfScaledTangentialEvaluation")


class IntegralMomentOfNormalNormalEvaluation(Functional):

    r"""
    \int_F (n^T tau n) p ds
    p \in Polynomials
    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        # scaling on the normal is ok because edge length then weights
        # the reference element quadrature appropriately
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        transform = ref_el.get_entity_transform(sd - 1, facet)
        pts = tuple(map(lambda p: tuple(transform(p)), Q.get_points()))
        weights = Q.get_weights()
        pt_dict = OrderedDict()
        for pt, wgt, phi in zip(pts, weights, P_at_qpts):
            pt_dict[pt] = [(wgt*phi*n[i], (i, )) for i in range(sd)]
        super().__init__(ref_el, (sd, ), pt_dict, {}, "IntegralMomentOfScaledNormalEvaluation")
