from FIAT.dual_set import DualSet
from FIAT.finite_element import CiarletElement
from FIAT.reference_element import UFCInterval
from FIAT.functional import PointEvaluation, IntegralMoment
from FIAT.quadrature import make_quadrature
from FIAT.polynomial_set import ONPolynomialSet
import numpy

ufcint = UFCInterval()


class SillyElementDualSet(DualSet):

    def __init__(self, cell, order):
        assert cell == ufcint and order == 3
        entity_ids = {0: {0: [0], 1: [1]},
                      1: {0: [2, 3]}}
        vertnodes = [PointEvaluation(ufcint, xx)
                     for xx in ufcint.vertices]
        Q = make_quadrature(ufcint, 3)
        # 1st integral moment node is integral(1*f(x)*dx)
        ones = numpy.asarray([1.0 for x in Q.pts])
        # 2nd integral moment node is integral(x*f(x)*dx)
        xs = numpy.asarray([x for (x,) in Q.pts])
        intnodes = [IntegralMoment(ufcint, Q, ones),
                    IntegralMoment(ufcint, Q, xs)]
        nodes = vertnodes + intnodes
        super(SillyElementDualSet, self).__init__(nodes, ufcint, entity_ids)


class SillyElement(CiarletElement):

    def __init__(self, cell, order):
        assert cell == ufcint and order == 3
        poly_set = ONPolynomialSet(ufcint, 3)
        dual_set = SillyElementDualSet(ufcint, 3)
        super(SillyElement, self).__init__(poly_set, dual_set, 3, 0)