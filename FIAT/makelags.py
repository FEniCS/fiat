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

from . import lagrange
from . import reference_element
import string
import numpy

lagclass = \
"""class Lagrange%s%d: public FiniteElement {
public:
  Lagrange%s%d():FiniteElement(%d,%d,%d,%d,%d,%s) {}
  virtual ~Lagrange%s%d(){}
};"""

def array_to_C_string( u ):
    x = [ str( a ) for a in u ]
    return "{ %s }" % ( string.join( x , " , " ) )

def matrix_to_array( mat , mat_name ):
    (num_rows,num_cols) = mat.shape

    # get C array of data
    u = numpy.ravel( numpy.transpose( mat ) )

    array_name = mat_name
    return \
"""static double %s[] = %s;""" % ( array_name , \
        array_to_C_string( u ) )

T = reference_element.DefaultTriangle()
shape = "Triangle"
for i in range(3,4):
    L = lagrange.Lagrange(T,i)
    nb = L.get_nodal_basis()
    vdm = nb.get_coeffs()
    array_name="Lagrange%s%dCoeffs"%(shape,i)
    print(matrix_to_array( vdm , array_name ))
    print(lagclass % (shape,i,shape,i,\
                          nb.get_degree(), \
                          nb.get_embedded_degree(), \
                          2,\
                          nb.get_num_members() , \
                          nb.get_num_members() , \
                          array_name,shape,i))

