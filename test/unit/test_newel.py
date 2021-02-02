from FIAT.newel import NewElement
from FIAT.reference_element import UFCInterval
import numpy

def test_newel():
    # can create
    el = NewElement(UFCInterval(), 3)
    # dual eval gives expected values from sum of point evaluations
    fns = (lambda x: x[0], lambda x: x[0]**2)
    expected = ([0, 1, 1/2, 1/3], [0, 1, 1/3, 1/4])
    for i in range(len(fns)):
        node_vals = []
        for node in el.dual.nodes:
            pt_dict = node.pt_dict
            node_val = 0.0
            for pt in pt_dict:
                for (w, _) in pt_dict[pt]:
                    node_val += w * fns[i](pt)
            node_vals.append(node_val)
        assert len(node_vals) == len(expected[i])
        assert numpy.allclose(node_vals, expected[i])
