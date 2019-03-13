import pytest
import numpy as np
from FIAT.reference_element import *
from FIAT.serendipity import *

#Check the right number of basis functions are being produced.

@pytest.mark.parametrize("dim", range(2,4))
def test_serendipity_creation(dim):
    from FIAT import ufc_cell, make_quadrature

    cell = [None, "interval", "quadrilateral", "hexahedron"]
    ref_el = ufc_cell(cell[dim])
    s = Serendipity(ref_el, 1)
    s_vals = s.tabulate(0,ref_el.get_vertices(),None)
    derivs = (0,)*dim
    assert np.sum(s_vals[derivs] - np.eye(len(ref_el.get_vertices()))) < 1e-14


def test_serendipity_basis():
    ref_el = UFCHexahedron()
    s = S(ref_el, 5)
    dim = ref_el.get_spatial_dimension()
    derivs = (0,)*dim
    assert len(s.basis[derivs]) == 74
    assert s.entity_ids[2][5][-1] == 73


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
