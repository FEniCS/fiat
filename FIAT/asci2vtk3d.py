# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.

#!/usr/bin/env python

# 3d mode: x y z f, f = f(x,y,z)
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]
    print(filename)
    base = filename.split(".")[0]
    output = "%s.vtk" % (base,)
    print("output to %s" % (output,))
else:
    print("python asci2vtk3d.py foo")
    sys.exit(0)


fin = open( filename , "r" )

coords = [ ]

for line in fin:
    coords.append( line.split() )

fin.close()

n = len( coords )

print("%s points" % (str(n),))


fout = open( output , "w" )
print("""# vtk DataFile Version 2.0
points
ASCII
DATASET UNSTRUCTURED_GRID
POINTS %s float""" % (str(n),), file=fout)

for c in coords:
    print("%s %s %s" % (c[0],c[1],c[2]), file=fout)

print("CELLS %s %s" % (n,2*n), file=fout)
for i in range( n ):
    print("1 %s" % (i,), file=fout)

print("CELL_TYPES %s" % (n,), file=fout)
for i in range( n ):
    print("1", file=fout)

print("POINT_DATA %s" % (n,), file=fout)
print("""SCALARS Z float 1
LOOKUP_TABLE default""", file=fout)

for i in range( n ):
    print(coords[i][3], file=fout)

fout.close()
