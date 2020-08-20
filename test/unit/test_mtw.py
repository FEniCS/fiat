import numpy as np
from FIAT import ufc_simplex, MardalTaiWinther, make_quadrature, expansions


def test_dofs():
    line = ufc_simplex(1)
    T = ufc_simplex(2)
    T.vertices = np.random.rand(3, 2)
    MTW = MardalTaiWinther(T, 3)

    Qline = make_quadrature(line, 6)

    linebfs = expansions.LineExpansionSet(line)
    linevals = linebfs.tabulate(1, Qline.pts)

    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        wts = np.asarray(Qline.wts)

        vals = MTW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        nvals = np.dot(np.transpose(vals, (0, 2, 1)), n)
        normal_moments = np.zeros((9, 2))
        for bf in range(9):
            for k in range(len(Qline.wts)):
                for m in (0, 1):
                    normal_moments[bf, m] += wts[k] * nvals[bf, k] * linevals[m, k]
        right = np.zeros((9, 2))
        right[3*ed, 0] = 1.0
        right[3*ed+2, 1] = 1.0
        assert np.allclose(normal_moments, right)
    for ed in range(3):
        t = T.compute_edge_tangent(ed)
        wts = np.asarray(Qline.wts)

        vals = MTW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        tvals = np.dot(np.transpose(vals, (0, 2, 1)), t)
        tangent_moments = np.zeros(9)
        for bf in range(9):
            for k in range(len(Qline.wts)):
                tangent_moments[bf] += wts[k] * tvals[bf, k] * linevals[0, k]
        right = np.zeros(9)
        right[3*ed + 1] = 1.0
        assert np.allclose(tangent_moments, right)
