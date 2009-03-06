# Written by Robert C. Kirby
# Copyright 2009 by Texas Tech University
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-07ER25821

import polynomial, numpy, shapes, BDFM, functional, quadrature

shape = shapes.TRIANGLE
n = 1
facet = 2

U = BDFM.BDFMSpace( shape , 1 )

pts = shapes.make_points( shape , 1 , facet , 3 )

normal = numpy.array( shapes.normals[shape][facet] )

u = U[0]

un = polynomial.Polynomial( U.base , numpy.dot( normal , \
                                                  u.dof ) )

Q = quadrature.make_quadrature( 1 , 4 )
Vline = polynomial.OrthogonalPolynomialSet( 1 , 1 )
un0trace = polynomial.projection(Vline, \
                                 lambda x:un((x[0],-1)) , \
                                 Q)
print un0trace.dof
