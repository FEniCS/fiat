import Nedelec, functional, Numeric, polynomial, quadrature, shapes

# returns ned, portion onto Pkm1 and portion onto PkH
#
def foo( k ):
    vec_Pkp1 = polynomial.OrthogonalPolynomialArraySet( 3 , k+1 )
    dimPkp1 = shapes.polynomial_dimension( 3 , k+1 )
    dimPk = shapes.polynomial_dimension( 3 , k )
    U = Nedelec.NedelecSpace( k )
    vec_Pk = vec_Pkp1.take( reduce( lambda a,b:a+b , \
                                  [ range(i*dimPkp1,i*dimPkp1+dimPk) \
                                    for i in range(3) ] ) )
    vec_Pk = vec_Pkp1.take( reduce( lambda a,b:a+b , \
                                  [ range(i*dimPkp1+dimPk,(i+1)*dimPkp1) \
                                    for i in range(3) ] ) )

    Q = quadrature.make_quadrature( 3 , 2 * (k + 1) )

    return U

                                        
for k in range(1):
    U = foo( k )
    pts = shapes.make_points( 3 , 1 , 5 , 3 )
    for u in U:
        for pt in pts:
            print Numeric.dot( pt , u( pt ) )
        print
    print
    print

    
