import Nedelec, functional, Numeric, polynomial, quadrature, shapes

# returns ned, portion onto Pkm1 and portion onto PkH
#
for k in range(3):
    U = Nedelec.NedelecSpace( k )
    print len( U )

    
