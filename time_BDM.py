import FIAT, FIAT.shapes, FIAT.BDM
import time,csv

instantiate_time = {}
tabulate_time = {}

reps = 10
deg_max = 8

shapes = (FIAT.shapes.TRIANGLE,FIAT.shapes.TETRAHEDRON)
shape_names = { FIAT.shapes.TRIANGLE:"Tri" , \
                FIAT.shapes.TETRAHEDRON:"Tet" }

for shape in shapes:
    instantiate_time[shape] = {}
    tabulate_time[shape] = {}
    for d in range(1,deg_max+1):
        time_cur = 0.0
        for rep in range(reps):
            t = time.time()
            U = FIAT.BDM.BDM(shape,d)
            time_cur += (time.time() - t)
        instantiate_time[shape][d] = time_cur / reps

        pts = FIAT.shapes.make_lattice( shape , d )

        time_cur = 0.0
        for rep in range(reps):
            t = time.time()
            U.function_space().tabulate(pts)
            time_cur += (time.time() - t)
        tabulate_time[shape][d] = time_cur / reps

f = open("instantiate_bdm_new.csv","w")
instantiate_csv = csv.writer(f)

# format should be
# blank , deg1 , deg2 , ... , degmax
# triangle , tdeg1 , tdeg2 , ... , tdegmax
# tetrahedron , tdeg1 , tdeg2 , ... , tdegmax
title = [ "Time to instantiate BDM elements" ]
instantiate_csv.writerow( title )
header = [ "" ] + range(1,deg_max+1)
instantiate_csv.writerow( header )
for shape in shapes:
    row_cur = [ shape_names[ shape ] ] \
              + instantiate_time[ shape ].values()
    instantiate_csv.writerow( row_cur )
f.close()

f = open("tabulate_bdm_new.csv","w")
tabulate_csv = csv.writer(f)
title = [ "Time to tabulate BDM elements at lattice points" ]
tabulate_csv.writerow( title )
header = [ "" ] + range(1,deg_max+1)
tabulate_csv.writerow( header )
for shape in shapes:
    row_cur = [ shape_names[ shape ] ] \
              + tabulate_time[ shape ].values()
    tabulate_csv.writerow( row_cur )
f.close()
