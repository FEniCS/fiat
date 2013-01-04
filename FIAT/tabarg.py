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

from . import argyris,reference_element

degree = 5
lattice_size = 10 * degree

T = reference_element.DefaultTriangle()

U = argyris.QuinticArgyris(T)
pts = T.make_lattice( lattice_size )

bfvals = U.get_nodal_basis().tabulate_new( pts )
u0 = bfvals[0]
fout = open("arg0.dat","w")
for i in range(len(pts)):
    fout.write("%s %s %s\n" % (pts[i][0],pts[i][1],u0[i]))
fout.close()

u1 = bfvals[1]
fout = open("arg1.dat","w")
for i in range(len(pts)):
    fout.write("%s %s %s\n" % (pts[i][0],pts[i][1],u1[i]))
fout.close()

u2 = bfvals[3]
fout = open("arg2.dat","w")
for i in range(len(pts)):
    fout.write("%s %s %s\n" % (pts[i][0],pts[i][1],u2[i]))
fout.close()

u3 = bfvals[18]
fout = open("arg3.dat","w")
for i in range(len(pts)):
    fout.write("%s %s %s\n" % (pts[i][0],pts[i][1],u3[i]))
fout.close()
