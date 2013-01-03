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

import hermite, reference_element, numpy

# Let's set up the reference triangle and another one
Khat = reference_element.UFCTriangle()

newverts = ((-1.0,0.0),(1.0,0.0),(0.0,1.0))
newtop = Khat.get_topology()

K = reference_element.ReferenceElement( reference_element.TRIANGLE , \
                                            newverts, \
                                            newtop )

# Construct the affine mapping between them
A,b = reference_element.make_affine_mapping( K.get_vertices() ,
                                             Khat.get_vertices() )

# build the Hermite element on the two triangles
Hhat = hermite.CubicHermite( Khat )
H = hermite.CubicHermite( K )

# get some points on each triangle
pts_hat = Khat.make_lattice( 6  )
pts = K.make_lattice( 6 )

# as a sanity check on the affine mapping, make sure
# pts map to pts_hat

for i in range( len( pts ) ):
    if not numpy.allclose( pts_hat[i],numpy.dot(A,pts[i]) + b):
        print("barf")

# Tabulate the Hermite basis on each triangle
Hhat_tabulated = Hhat.get_nodal_basis().tabulate_new( pts_hat )
H_tabulated = H.get_nodal_basis().tabulate_new( pts )

# transform:
M = numpy.zeros( (10,10),"d" )

Ainv = numpy.linalg.inv( A )

# entries for point values are easy
M[0,0] = 1.0
M[3,3] = 1.0
M[6,6] = 1.0
M[9,9] = 1.0
M[1:3,1:3] = numpy.transpose( Ainv )
M[4:6,4:6] = numpy.transpose( Ainv )
M[7:9,7:9] = numpy.transpose( Ainv )
# entries for rest are Jacobian


print(numpy.max( numpy.abs( H_tabulated - numpy.dot( numpy.transpose( M ) , Hhat_tabulated ) ) ))

