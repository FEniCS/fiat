import Nedelec, functional, Numeric, polynomial, quadrature, shapes
import RaviartThomas

for k in range(2,3):
    U = Nedelec.Nedelec( k )
