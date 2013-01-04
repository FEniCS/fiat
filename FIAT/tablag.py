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

from . import shapes, Lagrange

shape = 3
degree = 3
lattice_size = 10 * degree

U = Lagrange.Lagrange(shape,degree)
pts = shapes.make_lattice(shape,lattice_size)

us = U.function_space().tabulate(pts)

fout = open("foo.dat","w")
u0 = us[0]
for i in range(len(pts)):
    fout.write("%s %s %s %s" % (pts[i][0],pts[i][1],pts[i][2],u0[i]))
fout.close()
