# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Principal orthogonal expansion functions as defined by Karniadakis
and Sherwin.  These are parametrized over a reference element so as
to allow users to get coordinates that they want."""

import numpy
import math
import sympy
from FIAT import reference_element
from FIAT import jacobi


def jrc(a, b, n):
    an = (2*n+1+a+b)*(2*n+2+a+b) / (2*(n+1)*(n+1+a+b))
    bn = (a*a-b*b) * (2*n+1+a+b) / (2*(n+1)*(2*n+a+b)*(n+1+a+b))
    cn = (n+a)*(n+b)*(2*n+2+a+b) / ((n+1)*(n+1+a+b)*(2*n+a+b))
    return an, bn, cn


def _tabulate_dpts(tabulator, D, n, order, pts):
    X = sympy.DeferredVector('x')

    def form_derivative(F):
        '''Forms the derivative recursively, i.e.,
        F               -> [F_x, F_y, F_z],
        [F_x, F_y, F_z] -> [[F_xx, F_xy, F_xz],
                            [F_yx, F_yy, F_yz],
                            [F_zx, F_zy, F_zz]]
        and so forth.
        '''
        out = []
        try:
            out = [sympy.diff(F, X[j]) for j in range(D)]
        except (AttributeError, ValueError):
            # Intercept errors like
            #  AttributeError: 'list' object has no attribute
            #  'free_symbols'
            for f in F:
                out.append(form_derivative(f))
        return out

    def numpy_lambdify(X, F):
        '''Unfortunately, SymPy's own lambdify() doesn't work well with
        NumPy in that simple functions like
            lambda x: 1.0,
        when evaluated with NumPy arrays, return just "1.0" instead of
        an array of 1s with the same shape as x. This function does that.
        '''
        try:
            lambda_x = [numpy_lambdify(X, f) for f in F]
        except TypeError:  # 'function' object is not iterable
            # SymPy's lambdify also works on functions that return arrays.
            # However, use it componentwise here so we can add 0*x to each
            # component individually. This is necessary to maintain shapes
            # if evaluated with NumPy arrays.
            lmbd_tmp = sympy.lambdify(X, F)
            lambda_x = lambda x: lmbd_tmp(x) + 0 * x[0]
        return lambda_x

    def evaluate_lambda(lmbd, x):
        '''Properly evaluate lambda expressions recursively for iterables.
        '''
        try:
            values = [evaluate_lambda(l, x) for l in lmbd]
        except TypeError:  # 'function' object is not iterable
            values = lmbd(x)
        return values

    # Tabulate symbolically
    symbolic_tab = tabulator(n, X)
    # Make sure that the entries of symbolic_tab are lists so we can
    # append derivatives
    symbolic_tab = [[phi] for phi in symbolic_tab]
    #
    data = (order + 1) * [None]
    for r in range(order + 1):
        shape = [len(symbolic_tab), len(pts)] + r * [D]
        data[r] = numpy.empty(shape)
        for i, phi in enumerate(symbolic_tab):
            # Evaluate the function numerically using lambda expressions
            deriv_lambda = numpy_lambdify(X, phi[r])
            data[r][i] = \
                numpy.array(evaluate_lambda(deriv_lambda, pts.T)).T
            # Symbolically compute the next derivative.
            # This actually happens once too many here; never mind for
            # now.
            phi.append(form_derivative(phi[-1]))
    return data


def xi_triangle(eta):
    """Maps from [-1,1]^2 to the (-1,1) reference triangle."""
    eta1, eta2 = eta
    xi1 = 0.5 * (1.0 + eta1) * (1.0 - eta2) - 1.0
    xi2 = eta2
    return (xi1, xi2)


def xi_tetrahedron(eta):
    """Maps from [-1,1]^3 to the -1/1 reference tetrahedron."""
    eta1, eta2, eta3 = eta
    xi1 = 0.25 * (1. + eta1) * (1. - eta2) * (1. - eta3) - 1.
    xi2 = 0.5 * (1. + eta2) * (1. - eta3) - 1.
    xi3 = eta3
    return xi1, xi2, xi3


class PointExpansionSet(object):
    """Evaluates the point basis on a point reference element."""

    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 0:
            raise ValueError("Must have a point")
        self.ref_el = ref_el
        self.base_ref_el = reference_element.Point()

    def get_num_members(self, n):
        return 1

    def tabulate(self, n, pts):
        """Returns a numpy array A[i,j] = phi_i(pts[j]) = 1.0."""
        assert n == 0
        return numpy.ones((1, len(pts)))

    def tabulate_derivatives(self, n, pts):
        """Returns a numpy array of size A where A[i,j] = phi_i(pts[j])
        but where each element is an empty tuple (). This maintains
        compatibility with the interfaces of the interval, triangle and
        tetrahedron expansions."""
        deriv_vals = numpy.empty_like(self.tabulate(n, pts), dtype=tuple)
        deriv_vals.fill(())

        return deriv_vals


class LineExpansionSet(object):
    """Evaluates the Legendre basis on a line reference element."""

    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 1:
            raise Exception("Must have a line")
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultLine()
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A, self.b = reference_element.make_affine_mapping(v1, v2)
        self.mapping = lambda x: numpy.dot(self.A, x) + self.b
        self.scale = numpy.sqrt(numpy.linalg.det(self.A))

    def get_num_members(self, n):
        return n + 1

    def tabulate(self, n, pts):
        """Returns a numpy array A[i,j] = phi_i(pts[j])"""
        if len(pts) > 0:
            ref_pts = numpy.array([self.mapping(pt) for pt in pts])
            psitilde_as = jacobi.eval_jacobi_batch(0, 0, n, ref_pts)

            results = numpy.zeros((n + 1, len(pts)), type(pts[0][0]))
            for k in range(n + 1):
                results[k, :] = psitilde_as[k, :] * math.sqrt(k + 0.5)

            return results
        else:
            return []

    def tabulate_derivatives(self, n, pts):
        """Returns a tuple of length one (A,) such that
        A[i,j] = D phi_i(pts[j]).  The tuple is returned for
        compatibility with the interfaces of the triangle and
        tetrahedron expansions."""
        ref_pts = numpy.array([self.mapping(pt) for pt in pts])
        psitilde_as_derivs = jacobi.eval_jacobi_deriv_batch(0, 0, n, ref_pts)

        # Jacobi polynomials defined on [-1, 1], first derivatives need scaling
        psitilde_as_derivs *= 2.0 / self.ref_el.volume()

        results = numpy.zeros((n + 1, len(pts)), "d")
        for k in range(0, n + 1):
            results[k, :] = psitilde_as_derivs[k, :] * numpy.sqrt(k + 0.5)

        vals = self.tabulate(n, pts)
        deriv_vals = (results,)

        # Create the ordinary data structure.
        dv = []
        for i in range(vals.shape[0]):
            dv.append([])
            for j in range(vals.shape[1]):
                dv[-1].append((vals[i][j], [deriv_vals[0][i][j]]))

        return dv


class TriangleExpansionSet(object):
    """Evaluates the orthonormal Dubiner basis on a triangular
    reference element."""

    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 2:
            raise Exception("Must have a triangle")
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultTriangle()
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A, self.b = reference_element.make_affine_mapping(v1, v2)
        self.mapping = lambda x: numpy.dot(self.A, x) + self.b
#        self.scale = numpy.sqrt(numpy.linalg.det(self.A))

    def get_num_members(self, n):
        return (n + 1) * (n + 2) // 2

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        else:
            return numpy.array(self._tabulate(n, numpy.array(pts).T))

    def _tabulate(self, n, pts):
        '''A version of tabulate() that also works for a single point.
        '''
        m1, m2 = self.A.shape
        ref_pts = [sum(self.A[i][j] * pts[j] for j in range(m2)) + self.b[i]
                   for i in range(m1)]

        def idx(p, q):
            return (p + q) * (p + q + 1) // 2 + q

        results = ((n + 1) * (n + 2) // 2) * [None]

        results[0] = 1.0 \
            + pts[0] - pts[0] \
            + pts[1] - pts[1]

        if n == 0:
            return results

        x = ref_pts[0]
        y = ref_pts[1]

        f1 = (1.0 + 2 * x + y) / 2.0
        f2 = (1.0 - y) / 2.0
        f3 = f2**2

        results[idx(1, 0)] = f1

        for p in range(1, n):
            a = (2.0 * p + 1) / (1.0 + p)
            # b = p / (p+1.0)
            results[idx(p+1, 0)] = a * f1 * results[idx(p, 0)] \
                - p/(1.0+p) * f3 * results[idx(p-1, 0)]

        for p in range(n):
            results[idx(p, 1)] = 0.5 * (1+2.0*p+(3.0+2.0*p)*y) \
                * results[idx(p, 0)]

        for p in range(n - 1):
            for q in range(1, n - p):
                (a1, a2, a3) = jrc(2 * p + 1, 0, q)
                results[idx(p, q+1)] = \
                    (a1 * y + a2) * results[idx(p, q)] \
                    - a3 * results[idx(p, q-1)]

        for p in range(n + 1):
            for q in range(n - p + 1):
                results[idx(p, q)] *= math.sqrt((p + 0.5) * (p + q + 1.0))

        return results
        # return self.scale * results

    def tabulate_derivatives(self, n, pts):
        order = 1
        data = _tabulate_dpts(self._tabulate, 2, n, order, numpy.array(pts))
        # Put data in the required data structure, i.e.,
        # k-tuples which contain the value, and the k-1 derivatives
        # (gradient, Hessian, ...)
        m = data[0].shape[0]
        n = data[0].shape[1]
        data2 = [[tuple([data[r][i][j] for r in range(order+1)])
                  for j in range(n)]
                 for i in range(m)]
        return data2

    def tabulate_jet(self, n, pts, order=1):
        return _tabulate_dpts(self._tabulate, 2, n, order, numpy.array(pts))


class TetrahedronExpansionSet(object):
    """Collapsed orthonormal polynomial expanion on a tetrahedron."""

    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 3:
            raise Exception("Must be a tetrahedron")
        self.ref_el = ref_el
        self.base_ref_el = reference_element.DefaultTetrahedron()
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A, self.b = reference_element.make_affine_mapping(v1, v2)
        self.mapping = lambda x: numpy.dot(self.A, x) + self.b
        self.scale = numpy.sqrt(numpy.linalg.det(self.A))

    def get_num_members(self, n):
        return (n + 1) * (n + 2) * (n + 3) // 6

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        else:
            return numpy.array(self._tabulate(n, numpy.array(pts).T))

    def _tabulate(self, n, pts):
        '''A version of tabulate() that also works for a single point.
        '''
        m1, m2 = self.A.shape
        ref_pts = [sum(self.A[i][j] * pts[j] for j in range(m2)) + self.b[i]
                   for i in range(m1)]

        def idx(p, q, r):
            return (p + q + r)*(p + q + r + 1)*(p + q + r + 2)//6 + (q + r)*(q + r + 1)//2 + r

        results = ((n + 1) * (n + 2) * (n + 3) // 6) * [None]
        results[0] = 1.0 \
            + pts[0] - pts[0] \
            + pts[1] - pts[1] \
            + pts[2] - pts[2]

        if n == 0:
            return results

        x = ref_pts[0]
        y = ref_pts[1]
        z = ref_pts[2]

        factor1 = 0.5 * (2.0 + 2.0 * x + y + z)
        factor2 = (0.5 * (y + z))**2
        factor3 = 0.5 * (1 + 2.0 * y + z)
        factor4 = 0.5 * (1 - z)
        factor5 = factor4**2

        results[idx(1, 0, 0)] = factor1
        for p in range(1, n):
            a1 = (2.0 * p + 1.0) / (p + 1.0)
            a2 = p / (p + 1.0)
            results[idx(p+1, 0, 0)] = a1 * factor1 * results[idx(p, 0, 0)] \
                - a2 * factor2 * results[idx(p-1, 0, 0)]

        # q = 1
        for p in range(0, n):
            results[idx(p, 1, 0)] = results[idx(p, 0, 0)] \
                * (p * (1.0 + y) + (2.0 + 3.0 * y + z) / 2)

        for p in range(0, n - 1):
            for q in range(1, n - p):
                (aq, bq, cq) = jrc(2 * p + 1, 0, q)
                qmcoeff = aq * factor3 + bq * factor4
                qm1coeff = cq * factor5
                results[idx(p, q+1, 0)] = qmcoeff * results[idx(p, q, 0)] \
                    - qm1coeff * results[idx(p, q-1, 0)]

        # now handle r=1
        for p in range(n):
            for q in range(n - p):
                results[idx(p, q, 1)] = results[idx(p, q, 0)] \
                    * (1.0 + p + q + (2.0 + q + p) * z)

        # general r by recurrence
        for p in range(n - 1):
            for q in range(0, n - p - 1):
                for r in range(1, n - p - q):
                    ar, br, cr = jrc(2 * p + 2 * q + 2, 0, r)
                    results[idx(p, q, r+1)] = \
                        (ar * z + br) * results[idx(p, q, r)] \
                        - cr * results[idx(p, q, r-1)]

        for p in range(n + 1):
            for q in range(n - p + 1):
                for r in range(n - p - q + 1):
                    results[idx(p, q, r)] *= \
                        math.sqrt((p+0.5)*(p+q+1.0)*(p+q+r+1.5))

        return results

    def tabulate_derivatives(self, n, pts):
        order = 1
        D = 3
        data = _tabulate_dpts(self._tabulate, D, n, order, numpy.array(pts))
        # Put data in the required data structure, i.e.,
        # k-tuples which contain the value, and the k-1 derivatives
        # (gradient, Hessian, ...)
        m = data[0].shape[0]
        n = data[0].shape[1]
        data2 = [[tuple([data[r][i][j] for r in range(order + 1)])
                  for j in range(n)]
                 for i in range(m)]
        return data2

    def tabulate_jet(self, n, pts, order=1):
        return _tabulate_dpts(self._tabulate, 3, n, order, numpy.array(pts))


def get_expansion_set(ref_el):
    """Returns an ExpansionSet instance appopriate for the given
    reference element."""
    if ref_el.get_shape() == reference_element.POINT:
        return PointExpansionSet(ref_el)
    elif ref_el.get_shape() == reference_element.LINE:
        return LineExpansionSet(ref_el)
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return TriangleExpansionSet(ref_el)
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return TetrahedronExpansionSet(ref_el)
    else:
        raise Exception("Unknown reference element type.")


def polynomial_dimension(ref_el, degree):
    """Returns the dimension of the space of polynomials of degree no
    greater than degree on the reference element."""
    if ref_el.get_shape() == reference_element.POINT:
        if degree > 0:
            raise ValueError("Only degree zero polynomials supported on point elements.")
        return 1
    elif ref_el.get_shape() == reference_element.LINE:
        return max(0, degree + 1)
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return max((degree + 1) * (degree + 2) // 2, 0)
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return max(0, (degree + 1) * (degree + 2) * (degree + 3) // 6)
    else:
        raise ValueError("Unknown reference element type.")
