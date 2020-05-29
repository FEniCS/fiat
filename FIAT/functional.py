# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# Modified 2020 by the same from Baylor University
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

from FIAT import polynomial_set


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


class Functional(object):
    r"""Abstract class representing a linear functional.
    All FIAT functionals are discrete in the sense that
    they are written as a weighted sum of (derivatives of components of) their
    argument evaluated at particular points.

    :arg ref_el: a :class:`Cell`
    :arg target_shape: a tuple indicating the value shape of functions on
         the functional operates (e.g. if the function eats 2-vectors
         then target_shape is (2,) and if it eats scalars then
         target_shape is ()
    :arg pt_dict: A dict mapping points to lists of information about
         how the functional is evaluated.  Each entry in the list takes
         the form of a tuple (wt, comp) so that (at least if the
         deriv_dict argument is empty), the functional takes the form
         :math:`\ell(f) = \sum_{q=1}^{N_q} \sum_{k=1}^{K_q} w^q_k f_{c_k}(x_q)`
         where :math:`f_{c_k}` indicates a particular vector or tensor component
    :arg deriv_dict: A dict that is similar to `pt_dict`, although the entries
         of each list are tuples (wt, alpha, comp) with alpha a tuple
         of nonnegative integers corresponding to the order of partial
         differentiation in each spatial direction.
    :arg functional_type: a string labeling the kind of functional
         this is.
    """
    def __init__(self, ref_el, target_shape, pt_dict, deriv_dict,
                 functional_type):
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

    def to_riesz(self, poly_set):
        r"""Constructs an array representation of the functional so
        that the functional may be applied to a function expressed in
        in terms of the expansion set underlying  `poly_set` by means
        of contracting coefficients.

        That is, `poly_set` will have members all expressed in the
        form :math:`p = \sum_{i} \alpha^i \phi_i`
        where :math:`\{\phi_i\}_{i}` is some orthonormal expansion set
        and :math:`\alpha^i` are coefficients.  Note: the orthonormal
        expansion set is always scalar-valued but if the members of
        `poly_set` are vector or tensor valued the :math:`\alpha^i`
        will be scalars or vectors.

        This function constructs a tensor :math:`R` such that the
        contraction of :math:`R` with the array of coefficients
        :math:`\alpha` produces the effect of :math:`\ell(f)`

        In the case of scalar-value functions, :math:`R` is just a
        vector of the same length as the expansion set, and
        :math:`R_i = \ell(\phi_i)`.  For vector-valued spaces,
        :math:`R_{ij}` will be :math:`\ell(e^i \phi_j)` where
        :math:`e^i` is the canonical unit vector nonzero only in one
        entry :math:`i`.
        """
        es = poly_set.get_expansion_set()
        ed = poly_set.get_embedded_degree()
        nexp = es.get_num_members(ed)

        pt_dict = self.get_point_dict()

        pts = list(pt_dict.keys())
        npts = len(pts)

        bfs = es.tabulate(ed, pts)
        result = numpy.zeros(poly_set.coeffs.shape[1:], "d")

        # loop over points
        for j in range(npts):
            pt_cur = pts[j]
            wc_list = pt_dict[pt_cur]

            # loop over expansion functions
            for i in range(nexp):
                for (w, c) in wc_list:
                    result[c][i] += w * bfs[i, j]

        if self.deriv_dict:
            dpt_dict = self.deriv_dict

            # this makes things quicker since it uses dmats after
            # instantiation
            es_foo = polynomial_set.ONPolynomialSet(self.ref_el, ed)
            dpts = list(dpt_dict.keys())

            dbfs = es_foo.tabulate(dpts, self.max_deriv_order)

            ndpts = len(dpts)
            for j in range(ndpts):
                dpt_cur = dpts[j]
                wac_list = dpt_dict[dpt_cur]
                for i in range(nexp):
                    for (w, alpha, c) in wac_list:
                        result[c][i] += w * dbfs[tuple(alpha)][i, j]

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
        dpt_dict = {x: [(1.0, tuple(alpha), tuple())]}
        self.alpha = tuple(alpha)
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


class PointNormalDerivative(Functional):
    """Represents d/dn at a point on a facet."""
    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        sd = ref_el.get_spatial_dimension()

        alphas = []
        for i in range(sd):
            alpha = [0] * sd
            alpha[i] = 1
            alphas.append(alpha)
        dpt_dict = {pt: [(n[i], tuple(alphas[i]), tuple()) for i in range(sd)]}

        Functional.__init__(self, ref_el, tuple(), {}, dpt_dict, "PointNormalDeriv")


class PointNormalSecondDerivative(Functional):
    """Represents d^/dn^2 at a point on a facet."""
    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        sd = ref_el.get_spatial_dimension()
        tau = numpy.zeros((sd*(sd+1)//2,))

        alphas = []
        cur = 0
        for i in range(sd):
            for j in range(i, sd):
                alpha = [0] * sd
                alpha[i] += 1
                alpha[j] += 1
                alphas.append(tuple(alpha))
                tau[cur] = n[i]*n[j]
                cur += 1

        self.tau = tau
        self.alphas = alphas
        dpt_dict = {pt: [(n[i], alphas[i], tuple()) for i in range(sd)]}

        Functional.__init__(self, ref_el, tuple(), {}, dpt_dict, "PointNormalDeriv")


class IntegralMoment(Functional):
    """Functional representing integral of the input against some tabulated function f.

    :arg ref_el: a :class:`Cell`.
    :arg Q: a :class:`QuadratureRule`.
    :arg f_at_qpts: an array tabulating the function f at the quadrature
         points.
    :arg comp: Optional argument indicating that only a particular
         component of the input function should be integrated against f
    :arg shp: Optional argument giving the value shape of input functions.
    """

    def __init__(self, ref_el, Q, f_at_qpts, comp=tuple(), shp=tuple()):
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

        alphas = [tuple([1 if j == i else 0 for j in range(sd)]) for i in range(sd)]
        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*n[i]*f_at_qpts[j], alphas[i], tuple()) for i in range(sd)]

        Functional.__init__(self, ref_el, tuple(),
                            {}, dpt_dict, "IntegralMomentOfNormalDerivative")


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


class IntegralMomentOfDivergence(Functional):
    def __init__(self, ref_el, Q, f_at_qpts):
        self.f_at_qpts = f_at_qpts
        self.Q = Q

        sd = ref_el.get_spatial_dimension()

        qpts, qwts = Q.get_points(), Q.get_weights()
        dpts = qpts
        self.dpts = dpts

        dpt_dict = OrderedDict()

        alphas = [tuple([1 if j == i else 0 for j in range(sd)]) for i in range(sd)]
        for j, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[j]*f_at_qpts[j], alphas[i], (i,)) for i in range(sd)]

        Functional.__init__(self, ref_el, tuple(),
                            {}, dpt_dict, "IntegralMomentOfDivergence")


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

        alphas = [tuple([1 if j == i else 0 for j in range(sd)]) for i in range(sd)]
        for q, pt in enumerate(dpts):
            dpt_dict[tuple(pt)] = [(qwts[q]*f_at_qpts[i, q], alphas[j], (i, j)) for i in range(2) for j in range(2)]

        Functional.__init__(self, ref_el, tuple(),
                            {}, dpt_dict, "IntegralMomentOfTensorDivergence")
