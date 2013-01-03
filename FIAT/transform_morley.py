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

import morley, reference_element, numpy

# Let's set up the reference triangle and another one
Khat = reference_element.UFCTriangle()

newverts = ((-1.0,0.0),(42.0,-0.5),(0.0,1.0))
newtop = Khat.get_topology()

K = reference_element.ReferenceElement( reference_element.TRIANGLE , \
                                            newverts, \
                                            newtop )

# Construct the affine mapping between them
A,b = reference_element.make_affine_mapping( K.get_vertices() ,
                                             Khat.get_vertices() )

# build the Morley element on the two triangles
Mhat = morley.Morley( Khat )
M = morley.Morley( K )

# get some points on each triangle
pts_hat = Khat.make_lattice( 4 , 1 )
pts = K.make_lattice( 4 , 1 )

# as a sanity check on the affine mapping, make sure
# pts map to pts_hat

for i in range( len( pts ) ):
    if not numpy.allclose( pts_hat[i],numpy.dot(A,pts[i]) + b):
        print("barf")

# Tabulate the Morley basis on each triangle
Mhat_tabulated = Mhat.get_nodal_basis().tabulate_new( pts_hat )
M_tabulated = M.get_nodal_basis().tabulate_new( pts )

Ainv = numpy.linalg.inv( A )
AinvT = numpy.transpose( Ainv )

D = numpy.zeros( (6,9) , "d" )
E = numpy.zeros( (9,6) , "d" )

D[0,0] = 1.0
D[1,1] = 1.0
D[2,2] = 1.0

for i in range(3):
    n = K.compute_normal(i)
    t = K.compute_normalized_edge_tangent(i)
    nhat = Khat.compute_normal(i)
    l = K.volume_of_subcomplex(1,i)
    nt = numpy.transpose( [ n , t ] )
    [f,g] = numpy.dot( nhat , numpy.dot( AinvT , nt ) ) / l
    D[3+i,3+i] = f
    D[3+i,6+i] = g

for d in D.tolist():
    print(d)
print()

for i in range(3):
    E[i,i] = 1.0

for i in range(3):
    E[3+i,3+i] = K.volume_of_subcomplex(1,i)

for i in range(3):
    evids = K.topology[1][i]
    elen = K.volume_of_subcomplex( 1 , i )
    E[6+i,evids[1]] = 1.0
    E[6+i,evids[0]] = -1.0

print(E)
print()
transform = numpy.dot( D , E )
ttrans = numpy.transpose( transform )

for row in ttrans:
    print(row)
print()

print("max error")
print(numpy.max( numpy.abs( numpy.dot( numpy.transpose( transform ) , Mhat_tabulated )  - M_tabulated ) ))



