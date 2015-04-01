"""
    orthopoly.py - A suite of functions for generating orthogonal polynomials
    and quadrature rules.

    Copyright (c) 2014 Greg von Winckel 
    All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Last updated on Wed Jan  1 14:29:25 MST 2014
"""


from __future__ import division
import numpy as np
import scipy as sp
import scipy.linalg


def gauss(alpha,beta):
    """ 
    Compute the Gauss nodes and weights from the recursion 
    coefficients associated with a set of orthogonal polynomials 

    Inputs: 
    alpha - recursion coefficients
    beta - recursion coefficients

    Outputs: 
    x - quadrature nodes		
    w - quadrature weights

    Adapted from the MATLAB code by Walter Gautschi
    http://www.cs.purdue.edu/archives/2002/wxg/codes/gauss.m
    """
    
    from scipy.linalg import eig_banded

    A = np.vstack((np.sqrt(beta),alpha))
    x,V = eig_banded(A,lower=False) 
    w = beta[0]*sp.real(sp.power(V[0,:],2)) 
    return x,w


def radau(alpha,beta,xr):
    """
    Compute the Radau nodes and weights with the preassigned node xr
        
    Inputs: 
    alpha - recursion coefficients
    beta - recursion coefficients
    xr - assigned node location

    Outputs: 
    x - quadrature nodes		
    w - quadrature weights
   
    Based on the section 7 of the paper 
    "Some modified matrix eigenvalue problems" 
    by Gene Golub, SIAM Review Vol 15, No. 2, April 1973, pp.318--334
    """
    from scipy.linalg import solve_banded
    n = len(alpha)-1
    f = np.zeros(n)
    f[-1] = beta[-1]
    A = np.vstack((np.sqrt(beta),alpha-xr))
    J = np.vstack((A[:,0:-1],A[0,1:]))
    delta = solve_banded((1,1),J,f)
    alphar = alpha
    alphar[-1] = xr+delta[-1]  
    x,w = gauss(alphar,beta)
    return x,w
    

def lobatto(alpha,beta,xl1,xl2): 
    """
        Compute the Lobatto nodes and weights with the preassigned 
        nodea xl1,xl2
        
        Inputs: 
        alpha - recursion coefficients
        beta - recursion coefficients
        xl1 - assigned node location
        xl2 - assigned node location

        Outputs: 
        x - quadrature nodes        
        w - quadrature weights
   
        Based on the section 7 of the paper 
        "Some modified matrix eigenvalue problems" 
        by Gene Golub, SIAM Review Vol 15, No. 2, April 1973, pp.318--334
    """
    from scipy.linalg import solve_banded, solve
    n = len(alpha)-1
    en = np.zeros(n)
    en[-1] = 1
    A1 = np.vstack((np.sqrt(beta),alpha-xl1))
    J1 = np.vstack((A1[:,0:-1],A1[0,1:]))
    A2 = np.vstack((np.sqrt(beta),alpha-xl2))
    J2 = np.vstack((A2[:,0:-1],A2[0,1:]))
    g1 = solve_banded((1,1),J1,en)
    g2 = solve_banded((1,1),J2,en)
    C = np.array(((1,-g1[-1]),(1,-g2[-1])))
    xl = np.array((xl1,xl2))
    ab = solve(C,xl)
    
    alphal = alpha
    alphal[-1] = ab[0]  
    betal = beta
    betal[-1]=ab[1]
    x,w = gauss(alphal,betal)    
    return x,w
    

def rec_jacobi(N,a,b):
    """ 
    Generate the recursion coefficients alpha_k, beta_k 

    P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)
 
    for the Jacobi polynomials which are orthogonal on [-1,1] 
    with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b]  

    Inputs: 
    N - polynomial order
    a - weight parameter
    b - weight parameter
 
    Outputs: 
    alpha - recursion coefficients
    beta - recursion coefficients

    Adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
    http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m 
    """
    
    from scipy.special import gamma

    nu = (b-a)/float(a+b+2)
    mu = 2**(a+b+1)*gamma(a+1)*gamma(b+1)/gamma(a+b+2)
    
    if N == 1:
        alpha = nu
        beta = mu
    else:
        n = np.arange(1.0,N)
        nab =  2*n+a+b
        alpha = np.hstack((nu,(b**2-a**2)/(nab*(nab+2))))
        n = n[1:]
        nab = nab[1:]
        B1 = 4*(a+1)*(b+1)/float((a+b+2)**2*(a+b+3))
        B = 4*(n+a)*(n+b)*n*(n+a+b)/(nab**2*(nab+1)*(nab-1)) 
        beta = np.hstack((mu,B1,B))
    
    return alpha, beta



def rec_jacobi01(N,a,b):
    """
    Generate the recursion coefficients alpha_k, beta_k 
    for the Jacobi polynomials which are orthogonal on [0,1] 

    See rec_jacobi for the recursion coefficients on [-1,1]

    Inputs: 
    N - polynomial order
    a - weight parameter
    b - weight parameter
 
    Outputs: 
    alpha - recursion coefficients
    beta - recursion coefficients

    Adapted from the MATLAB implementation: 
    https://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi01.m

    """

    if a <= -1 or b <= -1:
        raise ValueError('''Jacobi coefficients are defined only 
                            for alpha,beta > -1''')

    if not isinstance(N,int):
        raise TypeError('N must be an integer')

    if N<1:
        raise ValueError('N must be at least 1')

    c,d = rec_jacobi(N,a,b)

    alpha = (1+c)/2 
    beta = d/4
    beta[0] = d[0]/2**(a+b+1)

    return alpha,beta


def polyval(alpha,beta,x):
    """ 
    Evaluate polynomials on x given the recursion coefficients alpha and beta 
    """
        
    N = len(alpha)
    m = len(x)
    P = np.zeros((m,N+1))
    
    P[:,0] = 1
    P[:,1] = (x-alpha[0])*P[:,0] 

    for k in xrange(1,N):
         P[:,k+1] = (x-alpha[k])*P[:,k] - beta[k]*P[:,k-1]

    return P  
    

def jacobi(N,a,b,x,NOPT=1):
    """ 
    JACOBI computes the Jacobi polynomials which are orthogonal on [-1,1] 
    with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b] and evaluate them 
    on the given grid up to P_N(x). Setting NOPT=2 returns the 
    L2-normalized polynomials 
    """
    
    m = len(x)
    P = np.zeros((m,N+1))

    apb = a+b
    a1 = a-1
    b1 = b-1
    c = apb*(a-b)

    P[:,0] = 1

    if N>0:
        P[:,1] = 0.5*(a-b+(apb+2)*x) 
     
    if N>1:
        for k in xrange(2,N+1):
            k2 = 2*k
            g = k2+apb
            g1 = g-1
            g2 = g-2
            d =  2.0*(k + a1)*(k + b1)*g
            P[:,k] = (g1*(c + g2*g*x)*P[:,k-1]-d*P[:,k-2])/(k2*(k + apb)*g2)

    if NOPT == 2:
        from scipy.special import gamma
        k = np.arange(N+1)
        pnorm = 2**(apb+1)*gamma(k+a+1)*gamma(k+b+1)/ \
                ((2*k+a+b+1)*(gamma(k+1)*gamma(k+a+b+1)))
        P *= 1/np.sqrt(pnorm) 
    return P


def jacobiD(N,a,b,x,NOPT=1):
    """ 
    JACOBID computes the first derivatives of the normalized Jacobi 
    polynomials which are orthogonal on [-1,1] with respect 
    to the weight w(x)=[(1-x)^a]*[(1+x)^b] and evaluate them 
    on the given grid up to P_N(x). Setting NOPT=2 returns
    the derivatives of the L2-normalized polynomials 
    """

    z = np.zeros((len(x),1))
    if  N == 0:
        Px = z
    else:
        
        Px = 0.5*np.hstack((z, jacobi(N-1,a+1,b+1,x,NOPT)* \
             ((a+b+2+np.arange(N)))))
    return Px



def mm_log(N,a):
    """ 
    MM_LOG Modified moments for a logarithmic weight function.
    
    The call mm=MM_LOG(n,a) computes the first n modified moments of the
    logarithmic weight function w(t)=t^a log(1/t) on [0,1] relative to 
    shifted Legendre polynomials. 

    REFERENCE:  Walter Gautschi,``On the preceding paper `A Legendre 
                polynomial integral' by James L. Blue'', 
                Math. Comp. 33 (1979), 742-743.

    Adapted from the MATLAB implementation: 
    https://www.cs.purdue.edu/archives/2002/wxg/codes/mm_log.m 
    """

    if a <= -1:
        raise ValueError('Parameter a must be greater than -1')

    prod = lambda z: reduce(lambda x,y:x*y,z,1)

    mm = np.zeros(N)

    c = 1
    for n in range(N):
        if isinstance(a,int) and a<n:

            p = range(n-a,n+a+2)
            mm[n] = (-1)**(n-a)/prod(p)
            mm[n] *= sp.special.gamma(a+1)**2
    
        else:
            if n == 0:
                mm[0] = 1/(a+1)**2
            else:
                k = np.arange(1,n+1)
                s = 1/(a+1+k)-1/(a+1-k)
                p = (a+1-k)/(a+1+k)
                mm[n] = (1/(a+1)+sum(s))*prod(p)/(a+1);

        mm[n] *= c
        c *= 0.5*(n+1)/(2*n+1)                 

    return mm


def mod_chebyshev(N,mom,alpham,betam):
    """
    Calcuate the recursion coefficients for the orthogonal polynomials
    which are are orthogonal with respect to a weight function which is
    represented in terms of its modifed moments which are obtained by
    integrating the monic polynomials against the weight function. 
         
    REFERENCES:
   
    John C. Wheeler, "Modified moments and Gaussian quadratures"
    Rocky Mountain Journal of Mathematics, Vol. 4, Num. 2 (1974), 287--296

    Walter Gautschi, "Orthogonal Polynomials (in Matlab)
    Journal of Computational and Applied Mathematics, Vol. 178 (2005) 215--234

    Adapted from the MATLAB implementation: 
    https://www.cs.purdue.edu/archives/2002/wxg/codes/chebyshev.m
    """

    if not isinstance(N,int):
        raise TypeError('N must be an integer')

    if N<1:
        raise ValueError('N must be at least 1')

    N = min(N,int(len(mom)/2))

    alpha = np.zeros(N)
    beta = np.zeros(N)
    normsq = np.zeros(N)
    sig = np.zeros((N+1,2*N))
    
    alpha[0] = alpham[0]+mom[1]/mom[0]
    beta[0] = mom[0]

    sig[1,:] = mom

    for n in range(2,N+1):
        for m in range(n-1,2*N-n+1):
            sig[n,m]=sig[n-1,m+1]-(alpha[n-2]-alpham[m])*sig[n-1,m] - \
                     beta[n-2]*sig[n-2,m]+betam[m]*sig[n-1,m-1]  
    
        alpha[n-1] = alpham[n-1]+sig[n,n]/sig[n,n-1]-sig[n-1,n-1]/ \
                     sig[n-1,n-2]
        beta[n-1] = sig[n,n-1]/sig[n-1,n-2]

    normsq = np.diagonal(sig,-1)

    return alpha,beta,normsq


def rec_jaclog(N,a):
    """ 
    Generate the recursion coefficients alpha_k, beta_k 

    P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)
 
    for the monic polynomials which are orthogonal on [0,1] 
    with respect to the weight w(x)=x^a*log(1/x)

    Inputs: 
    N - polynomial order
    a - weight parameter
 
    Outputs: 
    alpha - recursion coefficients
    beta - recursion coefficients

    Adated from the MATLAB code:
    https://www.cs.purdue.edu/archives/2002/wxg/codes/r_jaclog.m
    """
    alphaj,betaj = rec_jacobi01(2*N,0,0)
    mom = mm_log(2*N,a)
    alpha,beta,_ = mod_chebyshev(N,mom,alphaj,betaj)
    return alpha,beta
