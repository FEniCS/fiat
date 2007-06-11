import BDFM, shapes

shape = 2
degree = 2

U = BDFM.BDFM(shape,degree)

#pts = shapes.make_points(shape,shape,0,degree+3)
#pts = shapes.make_lattice( shape , 1 )

#print pts
#print U.function_space().tabulate( pts )

# check outward normal
for i in range(3):
    pts = shapes.make_points( shape , 1 , i , degree+2 )
    print pts
    vals = U.function_space().tabulate( pts )
    n = shapes.normals[shape][i]
    print n
    print vals[:,0,:]*n[0] + vals[:,1,:]*n[1]
    print

