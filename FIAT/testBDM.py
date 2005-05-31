import BDM, points, shapes, Numeric,time
shape = 2
kmax = 2
d = shapes.dimension(shape)

for k in range(1,kmax+1):
    t1 = time.time()
    bdm = BDM.BDM(shape,k)
    print time.time() - t1

# test: compute normal on edge/face 0 at control points.
# everything should be either zero or one.
##U = bdm.function_space()
##pts = points.make_points(shape,d-1,0,d+k)
##Utab = U.tabulate(pts)
##A = Numeric.zeros(Utab.shape[:2],"d")
##for i in range(Utab.shape[0]):
##    for j in range(Utab.shape[1]):
##        for k in range(d):
##            A[i,j] += Utab[i,j,k] / Numeric.sqrt(d)
##
##
##for i in range(A.shape[0]):
##    for j in range(A.shape[1]):
##        if abs(A[i,j]) > 1.e-12:
##            if abs(A[i,j]-1.0) > 1.e-12:
##                print "oops ", i, j
