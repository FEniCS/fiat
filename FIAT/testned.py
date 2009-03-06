# Written by Robert C. Kirby
# Copyright 2009 by Texas Tech University
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-07ER25821

import Nedelec, functional, numpy, polynomial, quadrature, shapes

#shape = shapes.TETRAHEDRON
shape = shapes.TRIANGLE
degree = 0

Ufs = Nedelec.Nedelec( shape , degree ).function_space()

pts = shapes.make_lattice( shape , 1 )

#print pts
#print Ufs.tabulate( pts ).shape
vals = Ufs.tabulate_jet( 1 , pts )

for v in vals:
    for u in v:
        for w in u:
            print w
            print u[w]
            print



#U = Uel.function_space()

#pts = shapes.make_points( shape , 1 , 2 , 3 )

#print pts
#print U.tabulate( pts )
