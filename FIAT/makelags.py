import lagrange
import reference_element
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
    print matrix_to_array( vdm , array_name )
    print lagclass % (shape,i,shape,i,\
                          nb.get_degree(), \
                          nb.get_embedded_degree(), \
                          2,\
                          nb.get_num_members() , \
                          nb.get_num_members() , \
                          array_name,shape,i)
    
