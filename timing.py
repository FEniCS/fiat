from FIAT import BDM,Lagrange,quadrature
import csv,time

deg_min = 1
deg_max = 8
reps = 4

els = {"Lagrange":Lagrange.Lagrange,\
       "BDM":BDM.BDMBulk1 }

shps = (2,3)
shp_names = {2:"Tri",3:"Tet"}

instantiate_time = {}
tabulate_time = {}

for el in els:
    print "Timing element %s" % (el,)
    instantiate_time[el] = {}
    tabulate_time[el] = {}
    for shape in shps:
        print "\tComputing on shape %s" % (shape,)
        instantiate_time[el][shape] = {}
        tabulate_time[el][shape] = {}
        for deg in range(deg_min,deg_max+1):
            print "\t\tComputing for degree %s" % (deg,)
            time_cur = 0
            print "\t\tInstantiating..."
            for i in range(reps):
                t1 = time.time()
                U = els[el](shape,deg)
                time_cur += ( time.time() - t1 )
            instantiate_time[el][shape][deg] = time_cur / reps
            print "\t\t\tTime: %s" % (instantiate_time[el][shape][deg],)
            print "\t\tTabulating..."
            qp = quadrature.make_quadrature(shape,deg)
            pts = qp.get_points()
            time_cur = 0
            for i in range(reps):
                t1 = time.time()
                U.function_space().tabulate(pts)
                time_cur += ( time.time() - t1 )
            tabulate_time[el][shape][deg] = time_cur / reps
            print "\t\t\tTime: %s" % (tabulate_time[el][shape][deg],)
f = open("time.csv","w")
time_csv = csv.writer( f )
time_csv.writerow( ["Instantiation"] )
time_csv.writerow( [""] + range(deg_min,deg_max+1) )
for el in els:
    for shape in shps:
        nm = "%s %s" % (el, shp_names[shape] )
        time_csv.writerow( [nm] + [ instantiate_time[el][shape][d]  \
                            for d in range(deg_min,deg_max+1) ] )
time_csv.writerow( ["Tabulation"] )
time_csv.writerow( [""] + range(deg_min,deg_max+1) )
for el in els:
    for shape in shps:
        nm = "%s %s" % (el, shp_names[shape] )
        time_csv.writerow( [nm] + [ tabulate_time[el][shape][d]  \
                            for d in range(deg_min,deg_max+1) ] )


f.close()

