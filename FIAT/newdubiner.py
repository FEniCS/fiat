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

import numpy

def jrc( a , b , n, num_type ):
    an = num_type( ( 2*n+1+a+b)*(2*n+2+a+b)) \
        / num_type( 2*(n+1)*(n+1+a+b) )
    bn = num_type( (a*a-b*b) * (2*n+1+a+b) ) \
        / num_type( 2*(n+1)*(2*n+a+b)*(n+1+a+b) )
    cn = num_type( (n+a)*(n+b)*(2*n+2+a+b) ) \
        / num_type( (n+1)*(n+1+a+b)*(2*n+a+b) )
    return an,bn,cn

def lattice_iter( start , finish , depth ):
    """Generator iterating over the depth-dimensional lattice of
    integers between start and (finish-1).  This works on simplices in
    1d, 2d, 3d, and beyond"""
    if depth == 0:
        return
    elif depth == 1:
        for ii in range( start , finish ):
            yield [ii]
    else:
        for ii in range( start , finish ):
            for jj in lattice_iter( start , finish-ii , depth - 1 ):
                yield [ii] + jj

def make_lattice( n , vs , numtype ):
    hs = numpy.array( [ ( vs[i] - vs[0] ) / numtype(n) \
                            for i in range(1,len(vs)) ] )
    
    result = []

    m = len(hs)
    for indices in lattice_iter(0,n+1,m):
        res_cur = vs[0].copy()
        for i in range(len(indices)):
            res_cur += indices[i] * hs[m-i-1]
        result.append( res_cur )

    return numpy.array( result )
    
def make_triangle_lattice( n , numtype ):
    vs = numpy.array( [ (numtype(-1) , numtype(-1)) , \
                            (numtype(1), numtype(-1)) , \
                            (numtype(-1), numtype(1)) ] )

    return make_lattice( n , vs , numtype )


def make_tetrahedron_lattice( n , numtype ):
    vs = numpy.array( [ (numtype(-1),numtype(-1),numtype(-1)) ,\
                        (numtype(1),numtype(-1),numtype(-1)),\
                        (numtype(-1),numtype(1),numtype(-1)),\
                        (numtype(-1),numtype(-1),numtype(1)) ] )
    return make_lattice( n , vs , numtype )

def make_lattice_dim( D , n , numtype ):
    if D == 2:
        return make_triangle_lattice( n , numtype )
    elif D == 3:
        return make_tetrahedron_lattice( n , numtype )

def tabulate_triangle( n , pts , numtype ):
    if len( pts ) == 0:
        return numpy.array( [] , numtype)

    def idx(p,q):
        return (p+q)*(p+q+1)/2 + q
    
    if numtype == float and type(pts[0][0]) == float:
        results = numpy.zeros( ( (n+1)*(n+2)/2,len(pts)), "d" )
    else:
        results = numpy.zeros( ( (n+1)*(n+2)/2,len(pts)), "O" )
    apts = numpy.array( pts )
    

    for ii in range( results.shape[1] ):
        results[0,ii] = numtype(1) + apts[ii,0]-apts[ii,0]+apts[ii,1]-apts[ii,1]
        
    if n == 0:
        return results
    
    x = apts[:,0]
    y = apts[:,1]

    one = numtype(1)
    two = numtype(2)
    three = numtype(3)

    foo = one + two *x + y

    f1 = (one+two*x+y)/two
    f2 = (one - y) / two
    f3 = f2**2

    results[idx(1,0),:] = f1

    for p in range(1,n):
        a = ( two * p + 1 ) / ( 1 + p )
        b = p / (p + one )
        results[idx(p+1,0)] = a * f1 * results[idx(p,0),:] \
            - p/(one+p) * f3 *results[idx(p-1,0),:]
        
    for p in range(n):
        results[idx(p,1),:] = (one + two*p+(three+two*p)*y)  / two\
            * results[idx(p,0)]
            
    for p in range(n-1):
        for q in range(1,n-p):
            (a1,a2,a3) = jrc(2*p+1,0,q,numtype)
            results[idx(p,q+1),:] \
                = ( a1 * y + a2 ) * results[idx(p,q)] \
                - a3 * results[idx(p,q-1)]

    return results


def tabulate_tetrahedron( n , pts , numtype ):
    def idx(p,q,r):
        return (p+q+r)*(p+q+r+1)*(p+q+r+2)/6 + (q+r)*(q+r+1)/2 + r
    
    if numtype == float and type(pts[0][0]) == float:
        tc = "d"
    else:
        tc = "O"
    apts = numpy.array( pts )

    results = numpy.zeros( ( (n+1)*(n+2)*(n+3)/6,len(pts)), tc)
    results[0,:] = 1.0 + apts[:,0]-apts[:,0]+apts[:,1]-apts[:,1]+apts[:,2]-apts[:,2]
        

    if n == 0:
        return results

    x = pts[:,0]
    y = pts[:,1]
    z = pts[:,2]

    one = numtype(1)
    two = numtype(2)
    three = numtype(3)

    factor1 = ( two + two*x + y + z ) / two
    factor2 = ((y+z)/two)**2
    factor3 = ( one + two * y + z ) / two
    factor4 = ( 1 - z ) / two
    factor5 = factor4 ** 2

    results[idx(1,0,0)] = factor1
    for p in range(1,n):
        a1 = ( two * p + one ) / ( p + one )
        a2 = p / (p + one)
        results[idx(p+1,0,0)] = a1 * factor1 * results[idx(p,0,0)] \
            -a2 * factor2 * results[ idx(p-1,0,0) ]

    for p in range(0,n):
        results[idx(p,1,0)] = results[idx(p,0,0)] \
            * ( p * (one + y) + ( two + three * y + z ) / two )

    for p in range(0,n-1):
        for q in range(1,n-p):
            (aq,bq,cq) = jrc(2*p+1,0,q,numtype)
            qmcoeff = aq * factor3 + bq * factor4
            qm1coeff = cq * factor5
            results[idx(p,q+1,0)] = qmcoeff * results[idx(p,q,0)] \
                - qm1coeff * results[idx(p,q-1,0)]

    for p in range(n):
        for q in range(n-p):
            results[idx(p,q,1)] = results[idx(p,q,0)] \
                * ( one + p + q + ( two + q + p ) * z )
    
    for p in range(n-1):
        for q in range(0,n-p-1):
            for r in range(1,n-p-q):
                ar,br,cr = jrc(2*p+2*q+2,0,r,numtype)
                results[idx(p,q,r+1)] = \
                    (ar * z + br) * results[idx(p,q,r) ] \
                    - cr * results[idx(p,q,r-1) ]
    

    return results


def tabulate( D , n , pts , numtype ):
    if D == 2:
        return tabulate_triangle( n , pts , numtype )
    elif D == 3:
        return tabulate_tetrahedron( n , pts , numtype )

def tabulate_jet( D , n , pts , order , numtype ):
    from Scientific.Functions.Derivatives import DerivVar as DV
    dpts = numpy.array( [ [ DV(pt[i],i,order) for i in range(len(pt)) ] \
                              for pt in pts ] )

    dbfs = tabulate( D , n , dpts , numtype )

    return dbfs



if __name__=="__main__":
    import gmpy
    from Scientific.Functions.Derivatives import DerivVar as DV

    latticeK = 2
    D = 3

    pts = make_tetrahedron_lattice( latticeK , gmpy.mpq )

    dpts = numpy.array( [ [ DV(pt[i],i) for i in range( len(pt) ) ] \
                              for pt in pts ] )

    vals = tabulate_tetrahedron( D , dpts , gmpy.mpq )

    print(vals)

