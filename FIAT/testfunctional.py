# Written by Robert C. Kirby
# Copyright 2009 by Texas Tech University
# Distributed under the LGPL license
# This work is partially supported by the US Department of Energy
# under award number DE-FG02-07ER25821

import functional,shapes,polynomial,Lagrange,functionalset, string, constrainedspaces

U = constrainedspaces.constrained_scalar_space( 2 , 2 , {1:{0:2,1:2,2:1},2:{0:2}} )

#pts = shapes.make_lattice( 2 , 2 )
pts = shapes.make_points( 2 , 1 , 2 , 5 )

print U.tabulate( pts )
