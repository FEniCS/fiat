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

"""Several functions related to the one-dimensional jacobi polynomials:
Evaluation, evaluation of derivatives, plus computation of the roots
via Newton's method.  These mainly are used in defining the expansion
functions over the simplices and in defining quadrature
rules over each domain."""

import math, numpy

def eval_jacobi(a,b,n,x):
    """Evaluates the nth jacobi polynomial with weight parameters a,b at a
    point x. Recurrence relations implemented from the pseudocode
    given in Karniadakis and Sherwin, Appendix B"""

    if 0 == n:
        return 1.0;
    elif 1 == n:
        return 0.5 * ( a - b + ( a + b + 2.0 ) * x )
    else: # 2 <= n
        apb = a + b
        pn2 = 1.0
        pn1 = 0.5 * ( a - b + ( apb + 2.0 ) * x )
        p = 0
        for k in range(2,n+1):
            a1 = 2.0 * k * ( k + apb ) * ( 2.0 * k + apb - 2.0 )
            a2 = ( 2.0 * k + apb - 1.0 ) * ( a * a - b * b )
            a3 = ( 2.0 * k + apb - 2.0 )  \
                 * ( 2.0 * k + apb - 1.0 ) \
                 * ( 2.0 * k + apb )
            a4 = 2.0 * ( k + a - 1.0 ) * ( k + b - 1.0 ) \
                 * ( 2.0 * k + apb )
            a2 = a2 / a1
            a3 = a3 / a1
            a4 = a4 / a1
            p = ( a2 + a3 * x ) * pn1 - a4 * pn2
            pn2 = pn1
            pn1 = p
        return p

def eval_jacobi_batch(a,b,n,xs):
    """Evaluates all jacobi polynomials with weights a,b
    up to degree n.  xs is a numpy.array of points.
    Returns a two-dimensional array of points, where the
    rows correspond to the Jacobi polynomials and the
    columns correspond to the points."""
    result = numpy.zeros( (n+1,len(xs)),xs.dtype )
    # hack to make sure AD type is propogated through
    for ii in range(result.shape[1]):
        result[0,ii] = 1.0 + xs[ii,0] - xs[ii,0]

    xsnew = xs.reshape((-1,))

    if n > 0:
        result[1,:] = 0.5 * ( a - b + ( a + b + 2.0 ) * xsnew )
    
        apb = a + b
        for k in range(2,n+1):
            a1 = 2.0 * k * ( k + apb ) * ( 2.0 * k + apb - 2.0 )
            a2 = ( 2.0 * k + apb - 1.0 ) * ( a * a - b * b )
            a3 = ( 2.0 * k + apb - 2.0 )  \
                * ( 2.0 * k + apb - 1.0 ) \
                * ( 2.0 * k + apb )
            a4 = 2.0 * ( k + a - 1.0 ) * ( k + b - 1.0 ) \
                * ( 2.0 * k + apb )
            a2 = a2 / a1
            a3 = a3 / a1
            a4 = a4 / a1
            result[k,:] = ( a2 + a3 * xsnew ) * result[k-1,:] \
                - a4 * result[k-2,:]
    return result

def eval_jacobi_deriv(a,b,n,x):
    """Evaluates the first derivative of P_{n}^{a,b} at a point x."""
    if n == 0:
        return 0.0
    else:
        return 0.5 * ( a + b + n + 1 ) * eval_jacobi(a+1,b+1,n-1,x)

def eval_jacobi_deriv_batch(a,b,n,xs):
    """Evaluates the first derivatives of all jacobi polynomials with
    weights a,b up to degree n.  xs is a numpy.array of points.
    Returns a two-dimensional array of points, where the
    rows correspond to the Jacobi polynomials and the
    columns correspond to the points."""
    results = numpy.zeros( (n+1,len(xs)), "d" )
    if n == 0:
        return results
    else:
        results[1:,:] = eval_jacobi_batch(a+1,b+1,n-1,xs)
    for j in range(1,n+1):
        results[j,:] *= 0.5*(a+b+j+1)
    return results


