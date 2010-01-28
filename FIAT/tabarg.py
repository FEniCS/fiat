import argyris,reference_element

degree = 5
lattice_size = 10 * degree

T = reference_element.DefaultTriangle()

U = argyris.QuinticArgyris(T)
pts = T.make_lattice( lattice_size )

bfvals = U.get_nodal_basis().tabulate_new( pts )
u0 = bfvals[0]
fout = open("arg0.dat","w")
for i in range(len(pts)):
    print >>fout, "%s %s %s" % (pts[i][0],pts[i][1],u0[i])
fout.close()

u1 = bfvals[1]
fout = open("arg1.dat","w")
for i in range(len(pts)):
    print >>fout, "%s %s %s" % (pts[i][0],pts[i][1],u1[i])
fout.close()

u2 = bfvals[3]
fout = open("arg2.dat","w")
for i in range(len(pts)):
    print >>fout, "%s %s %s" % (pts[i][0],pts[i][1],u2[i])
fout.close()

u3 = bfvals[18]
fout = open("arg3.dat","w")
for i in range(len(pts)):
    print >>fout, "%s %s %s" % (pts[i][0],pts[i][1],u3[i])
fout.close()
