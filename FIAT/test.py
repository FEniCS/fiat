import polynomial, Numeric, shapes, BDFM, functional, quadrature

shape = shapes.TRIANGLE
n = 1
facet = 2

U = BDFM.BDFMSpace( shape , 1 )

pts = shapes.make_points( shape , 1 , facet , 3 )

normal = Numeric.array( shapes.normals[shape][facet] )

u = U[0]

un = polynomial.Polynomial( U.base , Numeric.dot( normal , \
                                                  u.dof ) )

Q = quadrature.make_quadrature( 1 , 4 )
Vline = polynomial.OrthogonalPolynomialSet( 1 , 1 )
un0trace = polynomial.projection(Vline, \
                                 lambda x:un((x[0],-1)) , \
                                 Q)
print un0trace.dof
