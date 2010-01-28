#!/sw/bin/python

# 3d mode: x y z f, f = f(x,y,z)
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
    print filename
    base = filename.split(".")[0]
    output = "%s.vtk" % (base,)
    print "output to %s" % (output,)
else:
    print "python asci2vtk3d.py foo"
    sys.exit(0)


fin = open( filename , "r" )

coords = [ ]

for line in fin.xreadlines():
    coords.append( line.split() )

fin.close()

n = len( coords )

print "%s points" % (str(n),)


fout = open( output , "w" )
print >>fout, """# vtk DataFile Version 2.0
points
ASCII
DATASET UNSTRUCTURED_GRID
POINTS %s float""" % (str(n),)

for c in coords:
    print >>fout, "%s %s %s" % (c[0],c[1],c[2])

print >>fout, "CELLS %s %s" % (n,2*n)
for i in range( n ):
    print >>fout, "1 %s" % (i,)

print >>fout, "CELL_TYPES %s" % (n,)
for i in range( n ):
    print >>fout, "1"

print >>fout, "POINT_DATA %s" % (n,)
print >>fout, """SCALARS Z float 1
LOOKUP_TABLE default"""

for i in range( n ):
    print >>fout, coords[i][3]

fout.close()
