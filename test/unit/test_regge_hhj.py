from FIAT.reference_element import UFCTriangle
from FIAT import Regge, HellanHerrmannJohnson
import numpy as np
import pytest


def test_rotated_regge_is_hhj():
    triangle = UFCTriangle()

    R = Regge(triangle, 0)
    H = HellanHerrmannJohnson(triangle, 0)

    def S(u):
        return np.eye(2) * np.trace(u) - u

    for (r, h) in zip(R.tabulate(0, (0.2, 0.2))[(0, 0)],
                      H.tabulate(0, (0.2, 0.2))[(0, 0)]):
        assert np.all(np.isclose(r, S(h)))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
