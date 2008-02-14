"""
Entities dependent on the choice of reference element.
"""
import numpy

reference_element = "default"

# enumerated constants for shapes
LINE = 1
TRIANGLE = 2
TETRAHEDRON = 3

def get_vertices():
    """ Returns the vertices of the reference element """
    global reference_element
    if reference_element == "default":
        vertices = { LINE : { 0 : ( -1.0 , ) , 1 : ( 1.0 , ) } , \
                     TRIANGLE : { 0 : ( -1.0 , -1.0 ) , \
                                  1 : ( 1.0 , -1.0 ) , \
                                  2 : ( -1.0 , 1.0 ) } , \
                     TETRAHEDRON : { 0 : ( -1.0 , -1.0 , -1.0 ) , \
                                     1 : ( 1.0 , -1.0 , -1.0 ) , \
                                     2 : ( -1.0 , 1.0 , -1.0 ) , \
                                     3 : ( -1.0 , -1.0 , 1.0 ) } }
    elif reference_element == "UFC":
        vertices = { LINE : { 0 : ( 0.0 , ) , 1 : ( 1.0 , ) } , \
                     TRIANGLE : { 0 : ( 0.0 , 0.0 ) , \
                                  1 : ( 1.0 , 0.0 ) , \
                                  2 : ( 0.0 , 1.0 ) } , \
                     TETRAHEDRON : { 0 : ( 0.0 , 0.0 , 0.0 ) , \
                                     1 : ( 1.0 , 0.0 , 0.0 ) , \
                                     2 : ( 0.0 , 1.0 , 0.0 ) , \
                                     3 : ( 0.0 , 0.0 , 1.0 ) } }
    else:
        raise RuntimeError, "Unknown reference element: %s " \
              % str(reference_element)
    return vertices

def get_scale():
    if reference_element == "default":
        scale = 1.0 #Marie says scale = 2.0 no more
    elif reference_element == "UFC":
        scale = 1.0
    else:
        raise RuntimeError, "Unknown reference element", str(reference_element)
    return scale

def get_quadrature_weight_scale(shape): 
    if reference_element == "default":
        if shape == LINE: return 1.0
        if shape == TRIANGLE: return 0.5
        if shape == TETRAHEDRON: return 0.125
    elif reference_element == "UFC":
        if shape == LINE: return 0.5
        if shape == TRIANGLE: return 0.5*0.25
        if shape == TETRAHEDRON: return 0.125*0.5**3
    else:
        raise RuntimeError, "Unknown reference element", str(reference_element)

def get_chain_rule_scaling():
    global reference_element
    if reference_element == "default":
        return 1.0
    elif reference_element == "UFC":
        return 2.0
    else:
        raise RuntimeError, "Undefined reference element"
    
# These are coordinates changes mapping the [-1, 1] cube in d
# dimensions to the reference triangle. eta maps to cube to the
# triangle and xi maps the triangle to the cube and is thus the
# inverse of eta.  Definitions for the [-1, 1] case can be found on
# p.79-80 (Karniadakis & Sherwin)
def eta_line(xi):
    """ Maps from the reference element to the line [-1, 1]"""
    global reference_element
    (xi1,) = xi
    if reference_element == "default":
        eta1 = xi1 
    elif reference_element == "UFC":
        eta1 = 2.0*xi1-1.0
    else:
        raise RuntimeError, "Undefined reference element"
    return (eta1,)

def xi_line(eta):
    global reference_element
    (eta1,) = eta
    if reference_element == "default":
        xi1 = eta1
    elif reference_element == "UFC":
        xi1 = 0.5*(eta1+1.0)
    else:
        raise RuntimeError, "Undefined reference element"
    return (xi1,)

def eta_triangle( xi ):
    """ Maps from the reference element to the square [-1, 1]^2 """
    global reference_element
    (xi1, xi2) = xi
    if reference_element == "default":
        if xi2 == 1.0:
            eta1 = -1.0
        else:
            eta1 = 2.0*(1.0+xi1)/(1.0-xi2)-1
        eta2 = xi2 
    elif reference_element == "UFC":
        if xi2 == 1.0:
            eta1 = -1.0
        else:
            eta1 = 2.0*xi1/(1.0-xi2)-1
        eta2 = 2.0*xi2-1.0
    else:
        raise RuntimeError, "Undefined reference element"
    return eta1 , eta2

def xi_triangle( eta ):
    """ Maps from the square [-1, 1]^2 to the reference element"""
    global reference_element
    (eta1, eta2) = eta
    if reference_element == "default":
        xi1 = 0.5*(1.0+eta1)*(1.0-eta2)-1.0
        xi2 = eta2
    elif reference_element == "UFC":
        xi1 = 0.25*(eta1+1.0)*(1.0-eta2)
        xi2 = 0.5*(eta2+1.0)
    else:
        raise RuntimeError, "Undefined reference element"
    return xi1,xi2

def eta_tetrahedron( xi ):
    global reference_element
    if reference_element == "default":
        xi1,xi2,xi3 = xi
        if xi2 + xi3 == 0.:
            eta1 = 1.
        else:
            eta1 = -2. * ( 1. + xi1 ) / (xi2 + xi3) - 1.
        if xi3 == 1.:
            eta2 = -1.
        else:
            eta2 = 2. * (1. + xi2) / (1. - xi3 ) - 1.
        eta3 = xi3
    elif reference_element == "UFC":
        xi1,xi2,xi3 = xi
        if xi2 + xi3 == 1.:
            eta1 = 1.
        else:
            eta1 = -2.*xi1/(xi2 + xi3 - 1.) - 1.
        if xi3 == 1.:
            eta2 = -1.
        else:
            eta2 = 2.*xi2/(1.-xi3) - 1.
        eta3 = 2.0*xi3-1.0
    else:
        raise RuntimeError, "Undefined reference element"
    return eta1,eta2,eta3

def xi_tetrahedron( eta ):
    global reference_element
    eta1,eta2,eta3 = eta
    if reference_element == "default":
        xi1 = 0.25 * ( 1. + eta1 ) * ( 1. - eta2 ) * ( 1. - eta3 ) - 1.
        xi2 = 0.5 * ( 1. + eta2 ) * ( 1. - eta3 ) - 1.
        xi3 = eta3
    elif reference_element == "UFC":
        xi1 = 0.5*0.25*( 1. + eta1 )*( 1. - eta2 )*( 1. - eta3 )
        xi2 = 0.5*0.5*(1.+eta2)*(1.-eta3)
        xi3 = 0.5*(eta3+1.)
    else:
        raise RuntimeError, "Undefined reference element"

    return xi1,xi2,xi3



# The following four attributes (coord_changes, inverse_coord_changes,
# make_coordinate_change and make_inverse_coordinate_change) simply
# access eta and xi in a convenient way:
coord_changes = { TRIANGLE: eta_triangle , \
                  TETRAHEDRON: eta_tetrahedron }

inverse_coord_changes = { TRIANGLE: xi_triangle , \
                          TETRAHEDRON: xi_tetrahedron }

def make_coordinate_change( shape ):
    """Maps from reference domain to rectangular reference domain."""
    global coord_changes
    if coord_changes.has_key( shape ):
        return coord_changes[shape]
    else:
        raise RuntimeError, "Can't collapse coordinates"

def make_inverse_coordinate_change( shape ):
    """Maps from rectangular reference domain to reference domain."""
    global inverse_coord_changes
    if inverse_coord_changes.has_key( shape ):
        return inverse_coord_changes[shape]
    else:
        raise RuntimeError, "Can't collapse coordinates"


