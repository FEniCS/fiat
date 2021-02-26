# Copyright (C) 2021 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import numpy


def barycentric_interpolation(xsrc, xdst, order=0):
    """Return tabulations of a 1D Lagrange nodal basis via the second barycentric interpolation formula

    See Berrut and Trefethen (2004) https://doi.org/10.1137/S0036144502417715 Eq. (4.2) & (9.4)

    :arg xsrc: a :class:`numpy.array` with the nodes defining the Lagrange polynomial basis
    :arg xdst: a :class:`numpy.array` with the interpolation points
    :arg order: the integer order of differentiation
    :returns: dict of tabulations up to the given order (in the same format as :meth:`~.CiarletElement.tabulate`)
    """

    # w = barycentric weights
    # D = spectral differentiation matrix (D.T : u(xsrc) -> u'(xsrc))
    # I = barycentric interpolation matrix (I.T : u(xsrc) -> u(xdst))

    D = numpy.add.outer(-xsrc, xsrc)
    numpy.fill_diagonal(D, 1.0E0)
    w = 1.0E0 / numpy.prod(D, axis=0)
    D = numpy.divide.outer(w, w) / D
    numpy.fill_diagonal(D, numpy.diag(D) - numpy.sum(D, axis=0))

    I = numpy.add.outer(-xsrc, xdst)
    idx = numpy.argwhere(numpy.isclose(I, 0.0E0, 1E-14))
    I[idx[:, 0], idx[:, 1]] = 1.0E0
    I = 1.0E0 / I
    I *= w[:, None]
    I[:, idx[:, 1]] = 0.0E0
    I[idx[:, 0], idx[:, 1]] = 1.0E0
    I = (1.0E0 / numpy.sum(I, axis=0)) * I

    derivs = {(0,): I}
    for k in range(0, order):
        derivs[(k+1,)] = numpy.matmul(D, derivs[(k,)])

    return derivs
