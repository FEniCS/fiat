# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# last edited 9 May 2005

import shapes, functional, polynomial, points, functionalset

def PhiK( shape , n , U ):
    """The set PhiK defined by Brezzi & Fortin of vector-valued
    polynomials of degree n with vanishing divergence and
    vanishing normal components around the boundary."""
    U = polynomial.OrthogonalPolynomialArraySet( shape , n )
    # need Phi and U to share a common base, so I have to
    # take degree n and then throw away some to get degree n-1
    Phi = polynomial.OrthogonalPolynomialSet( shape , n )
    Phi_lower = Phi[:shapes.polynomial_dimension( shape , n-1 )]    
    DCPE = functional.DirectionalComponentPointEvaluation
    d = shapes.dimension( shape )
    pts_per_edge = [ [ x \
                       for x in points.make_points( shape , \
                                                    d-1 , \
                                                    i , \
                                                    n+d ) ] \
                    for i in shapes.entity_range( shape , d-1 ) ]
    pts_flat = reduce( lambda a,b:a+b , pts_per_edge )

    ls = [ functional.IntegralMomentOfDivergence( U , phi ) \
           for phi in Phi_lower ]

    for i in range(d+1):
        nrml = shapes.normals[ shape ][ i ]
        ls_cur = [ DCPE(U,nrml,pt) for pt in pts_per_edge[i] ]
        ls.extend(ls_cur)
    fset = functionalset.FunctionalSet(U,ls)

    return polynomial.ConstrainedPolynomialSet( fset )
