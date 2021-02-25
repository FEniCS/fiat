# This is work in progress

import numpy

def barycentric_interpolation(xsrc, xdst, order=0):
    """Returns a tuple with differentiation matrices up to a given order
    """

    # w = 1 / barycentric weights
    # D = spectral differentiation matrix
    # I = barycentric interpolation matrix

    D = numpy.subtract.outer(xsrc, xsrc)
    numpy.fill_diagonal(D, 1.0E0)
    w = numpy.prod(D, axis=1)

    D = numpy.divide.outer(w, w) / D
    numpy.fill_diagonal(D, numpy.diag(D) - numpy.sum(D, axis=1))

    I = numpy.subtract.outer(xdst, xsrc)
    idx = numpy.argwhere(numpy.isclose(I, 0.0E0, 1E-14))
    I[idx[:,0], idx[:,1]] = 1.0E0
    I = (1.0E0 / w) / I
    I[idx[:,0], :] = 0.0E0
    I[idx[:,0], idx[:,1]] = 1.0E0
    I *= (1.0E0/numpy.sum(I, axis=1))[:,None]
    
    derivs = {(0,) : I}
    for k in range(0, order):
        derivs[(k+1,)] = numpy.matmul(derivs[(k,)], D)

    for key in derivs:
        derivs[key] = derivs[key].T  # FIXME transpose
    return derivs
