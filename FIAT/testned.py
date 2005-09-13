import Nedelec, functional, Numeric, polynomial, quadrature, shapes
import RaviartThomas

for k in range(1):
#    U = RaviartThomas.RaviartThomas( 3 , k )
    U = Nedelec.NedelecSpace( k )
    Ud = Nedelec.NedelecDual( U , k )
    V = polynomial.FiniteElement( Ud , U )
    print V
