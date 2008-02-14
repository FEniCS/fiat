# Written by Robert C. Kirby
# Copyright 2005 by The University of Chicago
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-04ER25650

# Modified 26 Sept 2005
# Written 23 Sept 2005

import shapes, polynomial, functional, functionalset


def constrained_scalar_space( shape , max_order , orders ):
    """creates a scalar space of polynomials of degree max_order over shape.
    orders is a dictionary mapping dimensions to dictionaries mapping entities
    to polynomial degree to which each member is restricted along that edge.
    e.g. constrained_scalar_space(TRIANGLE,3,{1:{0:2,1:3,2:3},2:{0:3}})
    creates a space of polynomials that are cubic but whose members are
    only quadratic on edge 0 of the reference element.
    orders may be sparse and only contain entities on which the functions
    are constrained below max_order"""
    d = shapes.dimension( shape )

    U = polynomial.OrthogonalPolynomialSet( shape , max_order )

    constraints = []

    for i in orders:
        Uref = polynomial.OrthogonalPolynomialSet( i , max_order )
        orders_cur = orders[ i ]
        for j in orders_cur:
            order_cur = orders_cur[ j ]
        dim_P_order_cur = shapes.polynomial_dimension( i , order_cur )
        dim_P_max = shapes.polynomial_dimension( i , max_order )
        constraints += [ functional.FacetMoment( U , shape , i , j , Uref[k] ) \
                         for k in range( dim_P_order_cur , dim_P_max ) ]

    fset = functionalset.FunctionalSet( U , constraints )

    return polynomial.ConstrainedPolynomialSet( fset )



