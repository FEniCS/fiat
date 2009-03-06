# Written by Robert C. Kirby
# Copyright 2009 by Texas Tech University
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-07ER25821

import polynomial, shapes , functionalset, functional

def div_free( shape , n ):
    U = polynomial.OrthogonalPolynomialArraySet( shape , n )
    # need Phi and U to share a common base, so I have to
    # take degree n and then throw away some to get degree n-1
    Phi = polynomial.OrthogonalPolynomialSet( shape , n )
    Phi_lower = Phi[:shapes.polynomial_dimension( shape , n-1 )]

    ls = [ functional.IntegralMomentOfDivergence( U , phi ) \
           for phi in Phi_lower ]

    fset = functionalset.FunctionalSet(U,ls)

    return polynomial.ConstrainedPolynomialSet( fset )


def div_free_no_normal( shape , n ):
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

##    print len(pts_flat)
##    print len(Phi_lower)
##    print len(U)

    fset = functionalset.FunctionalSet(U,ls)

    return polynomial.ConstrainedPolynomialSet( fset )

shape = 2
degree = 2

Udivfree = div_free_no_normal( shape , degree )
U = polynomial.OrthogonalPolynomialArraySet( shape , degree )


#print len( Udivfree )
#print max( [ max(abs(polynomial.divergence(u).dof)) for u in Udivfree ] ) 
