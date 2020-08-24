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
from FIAT.quadrature import GaussLegendreQuadratureLineRule, QuadratureRule
from FIAT.reference_element import UFCInterval as interval


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
            alphas = [tuple(foo[1]) for foo in per_point]
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


class IntegralLegendreDirectionalMoment(Functional):
    """Moment of v.s against a Legendre polynomial over an edge"""
    def __init__(self, cell, s, entity, mom_deg, comp_deg, nm=""):
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

        super().__init__(cell, shp, pt_dict, {}, nm)


class IntegralLegendreNormalMoment(IntegralLegendreDirectionalMoment):
    """Moment of v.n against a Legendre polynomial over an edge"""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_scaled_normal(entity)
        super().__init__(cell, n, entity, mom_deg, comp_deg,
                         "IntegralLegendreNormalMoment")


class IntegralLegendreTangentialMoment(IntegralLegendreDirectionalMoment):
    """Moment of v.t against a Legendre polynomial over an edge"""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        t = cell.compute_edge_tangent(entity)
        super().__init__(cell, t, entity, mom_deg, comp_deg,
                         "IntegralLegendreTangentialMoment")


class IntegralLegendreBidirectionalMoment(Functional):
    """Moment of dot(s1, dot(tau, s2)) against Legendre on entity, multiplied by the size of the reference facet"""
    def __init__(self, cell, s1, s2, entity, mom_deg, comp_deg, nm=""):
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

        super().__init__(cell, shp, pt_dict, {}, nm)


class IntegralLegendreNormalNormalMoment(IntegralLegendreBidirectionalMoment):
    """Moment of dot(n, dot(tau, n)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_normal(entity)
        super().__init__(cell, n, n, entity, mom_deg, comp_deg,
                         "IntegralNormalNormalLegendreMoment")


class IntegralLegendreNormalTangentialMoment(IntegralLegendreBidirectionalMoment):
    """Moment of dot(n, dot(tau, t)) against Legendre on entity."""
    def __init__(self, cell, entity, mom_deg, comp_deg):
        n = cell.compute_normal(entity)
        t = cell.compute_normalized_edge_tangent(entity)
        super().__init__(cell, n, t, entity, mom_deg, comp_deg,
                         "IntegralNormalTangentialLegendreMoment")


class IntegralMomentOfDivergence(Functional):
    """Functional representing integral of the divergence of the input
    against some tabulated function f."""
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

        super().__init__(ref_el, tuple(), {}, dpt_dict,
                         "IntegralMomentOfDivergence")


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

        super().__init__(ref_el, tuple(), {}, dpt_dict,
                         "IntegralMomentOfDivergence")


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

        super().__init__(ref_el, shp, pt_dict, {}, "FrobeniusIntegralMoment")


class PointNormalEvaluation(Functional):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1."""

    def __init__(self, ref_el, facet_no, pt):
        n = ref_el.compute_normal(facet_no)
        self.n = n
        sd = ref_el.get_spatial_dimension()

        pt_dict = {pt: [(n[i], (i,)) for i in range(sd)]}

        shp = (sd,)
        super().__init__(ref_el, shp, pt_dict, {}, "PointNormalEval")


class PointEdgeTangentEvaluation(Functional):
    """Implements the evaluation of the tangential component of a
    vector at a point on a facet of dimension 1."""

    def __init__(self, ref_el, edge_no, pt):
        t = ref_el.compute_edge_tangent(edge_no)
        self.t = t
        sd = ref_el.get_spatial_dimension()
        pt_dict = {pt: [(t[i], (i,)) for i in range(sd)]}
        shp = (sd,)
        super().__init__(ref_el, shp, pt_dict, {}, "PointEdgeTangent")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.t)(%s)" % (','.join(x),)


class IntegralMomentOfEdgeTangentEvaluation(Functional):
    r"""
    \int_e v\cdot t p ds

    p \in Polynomials

    :arg ref_el: reference element for which e is a dim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg edge: which edge.
    """
    def __init__(self, ref_el, Q, P_at_qpts, edge):
        t = ref_el.compute_edge_tangent(edge)
        sd = ref_el.get_spatial_dimension()
        transform = ref_el.get_entity_transform(1, edge)
        pts = tuple(map(lambda p: tuple(transform(p)), Q.get_points()))
        weights = Q.get_weights()
        pt_dict = OrderedDict()
        for pt, wgt, phi in zip(pts, weights, P_at_qpts):
            pt_dict[pt] = [(wgt*phi*t[i], (i, )) for i in range(sd)]
        super().__init__(ref_el, (sd, ), pt_dict, {},
                         "IntegralMomentOfEdgeTangentEvaluation")


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


class IntegralMomentOfFaceTangentEvaluation(Functional):
    r"""
    \int_F v \times n \cdot p ds

    p \in Polynomials

    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        P_at_qpts = [[P_at_qpts[0][i], P_at_qpts[1][i], P_at_qpts[2][i]]
                     for i in range(P_at_qpts.shape[1])]
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        transform = ref_el.get_entity_transform(sd-1, facet)
        pts = tuple(map(lambda p: tuple(transform(p)), Q.get_points()))
        weights = Q.get_weights()
        pt_dict = OrderedDict()
        for pt, wgt, phi in zip(pts, weights, P_at_qpts):
            phixn = [phi[1]*n[2] - phi[2]*n[1],
                     phi[2]*n[0] - phi[0]*n[2],
                     phi[0]*n[1] - phi[1]*n[0]]
            pt_dict[pt] = [(wgt*(-n[2]*phixn[1]+n[1]*phixn[2]), (0, )),
                           (wgt*(n[2]*phixn[0]-n[0]*phixn[2]), (1, )),
                           (wgt*(-n[1]*phixn[0]+n[0]*phixn[1]), (2, ))]
        super().__init__(ref_el, (sd, ), pt_dict, {},
                         "IntegralMomentOfFaceTangentEvaluation")


class MonkIntegralMoment(Functional):
    r"""
    face nodes are \int_F v\cdot p dA where p \in P_{q-2}(f)^3 with p \cdot n = 0
    (cmp. Peter Monk - Finite Element Methods for Maxwell's equations p. 129)
    Note that we don't scale by the area of the facet

    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """

    def __init__(self, ref_el, Q, P_at_qpts, facet):
        sd = ref_el.get_spatial_dimension()
        weights = Q.get_weights()
        pt_dict = OrderedDict()
        transform = ref_el.get_entity_transform(sd-1, facet)
        pts = tuple(map(lambda p: tuple(transform(p)), Q.get_points()))
        for pt, wgt, phi in zip(pts, weights, P_at_qpts):
            pt_dict[pt] = [(wgt*phi[i], (i, )) for i in range(sd)]
        super().__init__(ref_el, (sd, ), pt_dict, {}, "MonkIntegralMoment")


class PointScaledNormalEvaluation(Functional):
    """Implements the evaluation of the normal component of a vector at a
    point on a facet of codimension 1, where the normal is scaled by
    the volume of that facet."""

    def __init__(self, ref_el, facet_no, pt):
        self.n = ref_el.compute_scaled_normal(facet_no)
        sd = ref_el.get_spatial_dimension()
        shp = (sd,)

        pt_dict = {pt: [(self.n[i], (i,)) for i in range(sd)]}
        super().__init__(ref_el, shp, pt_dict, {}, "PointScaledNormalEval")

    def tostr(self):
        x = list(map(str, list(self.pt_dict.keys())[0]))
        return "(u.n)(%s)" % (','.join(x),)


class IntegralMomentOfScaledNormalEvaluation(Functional):
    r"""
    \int_F v\cdot n p ds

    p \in Polynomials

    :arg ref_el: reference element for which F is a codim-1 entity
    :arg Q: quadrature rule on the face
    :arg P_at_qpts: polynomials evaluated at quad points
    :arg facet: which facet.
    """
    def __init__(self, ref_el, Q, P_at_qpts, facet):
        n = ref_el.compute_scaled_normal(facet)
        sd = ref_el.get_spatial_dimension()
        transform = ref_el.get_entity_transform(sd - 1, facet)
        pts = tuple(map(lambda p: tuple(transform(p)), Q.get_points()))
        weights = Q.get_weights()
        pt_dict = OrderedDict()
        for pt, wgt, phi in zip(pts, weights, P_at_qpts):
            pt_dict[pt] = [(wgt*phi*n[i], (i, )) for i in range(sd)]
        super().__init__(ref_el, (sd, ), pt_dict, {}, "IntegralMomentOfScaledNormalEvaluation")


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
        super().__init__(ref_el, shp, pt_dict, {}, "PointwiseInnerProductEval")


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
        super().__init__(ref_el, shp, pt_dict, {}, "TensorBidirectionalMomentInnerProductEvaluation")


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
