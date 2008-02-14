import functional,shapes,polynomial,Lagrange,functionalset, string, constrainedspaces

U = constrainedspaces.constrained_scalar_space( 2 , 2 , {1:{0:2,1:2,2:1},2:{0:2}} )

#pts = shapes.make_lattice( 2 , 2 )
pts = shapes.make_points( 2 , 1 , 2 , 5 )

print U.tabulate( pts )
