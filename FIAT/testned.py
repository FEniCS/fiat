import Nedelec, functional, numpy, polynomial, quadrature, shapes

shape = shapes.TETRAHEDRON
degree = 0

Uel = Nedelec.Nedelec( shape , degree )

U = Uel.function_space()

pts = shapes.make_points( shape , 1 , 2 , 3 )

print pts
print U.tabulate( pts )
