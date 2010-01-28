from FIAT import shapes, Lagrange

shape = 3
degree = 3
lattice_size = 10 * degree

U = Lagrange.Lagrange(shape,degree)
pts = shapes.make_lattice(shape,lattice_size)

us = U.function_space().tabulate(pts)

fout = open("foo.dat","w")
u0 = us[0]
for i in range(len(pts)):
    print >>fout, "%s %s %s %s" % (pts[i][0],pts[i][1],pts[i][2],u0[i])
fout.close()
