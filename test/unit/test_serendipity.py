from FIAT.reference_element import (
    UFCQuadrilateral, UFCInterval, TensorProductCell)
from FIAT import Serendipity
import numpy as np
import sympy


def test_serendipity_derivatives():
    cell = UFCQuadrilateral()
    S = Serendipity(cell, 2)

    x = sympy.DeferredVector("X")
    X, Y = x[0], x[1]
    basis_functions = [
        (1 - X)*(1 - Y),
        Y*(1 - X),
        X*(1 - Y),
        X*Y,
        Y*(1 - X)*(Y - 1),
        X*Y*(Y - 1),
        X*(1 - Y)*(X - 1),
        X*Y*(X - 1),
    ]
    points = [[0.5, 0.5], [0.25, 0.75]]
    for alpha, actual in S.tabulate(2, points).items():
        expect = list(sympy.diff(basis, *zip([X, Y], alpha))
                      for basis in basis_functions)
        expect = list([basis.subs(dict(zip([X, Y], point)))
                       for point in points]
                      for basis in expect)
        assert actual.shape == (8, 2)
        assert np.allclose(np.asarray(expect, dtype=float),
                           actual.reshape(8, 2))


def test_dual_tensor_versus_ufc():
    K0 = UFCQuadrilateral()
    ell = UFCInterval()
    K1 = TensorProductCell(ell, ell)
    S0 = Serendipity(K0, 2)
    S1 = Serendipity(K1, 2)
    # since both elements go through the flattened cell to produce the
    # dual basis, they ought to do *exactly* the same calculations and
    # hence form exactly the same nodes.
    for i in range(S0.space_dimension()):
        assert S0.dual.nodes[i].pt_dict == S1.dual.nodes[i].pt_dict
