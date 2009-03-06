# Written by Robert C. Kirby
# Copyright 2009 by Texas Tech University
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-07ER25821

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
    result = numpy.zeros( (n+1,len(xs)),"d" )
    result[0,:] = 1.0

    if n > 0:
	result[1,:] = 0.5 * ( a - b + ( a + b + 2.0 ) * xs )
	
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
	    result[k,:] = ( a2 + a3 * xs ) * result[k-1,:] \
		    - a4 * result[k-2,:]
    return result

def eval_jacobi_deriv(a,b,n,x):
    """Evaluates the first derivative of P_{n}^{a,b} at a point x."""
    if n == 0:
        return 0.0
    else:
        return 0.5 * ( a + b + n + 1 ) * eval_jacobi(a+1,b+1,n-1,x)

def eval_jacobi_deriv_batch(a,b,n,xs):
    results = numpy.zeros( (n+1,len(xs)), "d" )
    if n == 0:
	return results
    else:
	results[1:,:] = eval_jacobi_batch(a+1,b+1,n-1,xs)
	for j in range(1,n+1):
	    results[j,:] *= 0.5*(a+b+j+1)
	return results

def compute_gauss_jacobi_points( a , b , m ):
    """Computes the m roots of P_{m}^{a,b} on [-1,1] by Newton's method.
    The initial guesses are the Chebyshev points.  Algorithm
    implemented in Python from the pseudocode given by Karniadakis and
    Sherwin"""
    x = []
    eps = 1.e-8
    max_iter = 100
    for k in range(0,m):
        r = -math.cos(( 2.0*k + 1.0) * math.pi / ( 2.0 * m ) )
        if k > 0:
            r = 0.5 * ( r + x[k-1] ) 
        j = 0
        delta = 2 * eps
        while j < max_iter:
            s = 0
            for i in range(0,k):
                s = s + 1.0 / ( r - x[i] )
            f = eval_jacobi(a,b,m,r)
            fp = eval_jacobi_deriv(a,b,m,r)
            delta = f / (fp - f * s)
            
            r = r - delta
            
            if math.fabs(delta) < eps:
                break
            else:
                j = j + 1

        x.append(r)
    return x

