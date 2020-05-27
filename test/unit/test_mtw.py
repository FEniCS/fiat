import numpy as np


def test_dofs():
    from FIAT import ufc_simplex, MardalTaiWinther, make_quadrature, expansions

    line = ufc_simplex(1)
    T = ufc_simplex(2)
    T.vertices = np.random.rand(3, 2)
    MTW = MardalTaiWinther(T, 3)

    Qline = make_quadrature(line, 6)

    linebfs = expansions.LineExpansionSet(line)
    linevals = linebfs.tabulate(1, Qline.pts)

    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        print(n)
        wts = np.asarray(Qline.wts)

        vals = MTW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        nvals = np.dot(np.transpose(vals, (0, 2, 1)), n)
        normal_moments = np.zeros((9, 2))
        for bf in range(9):
            for k in range(len(Qline.wts)):
                for m in (0, 1):
                    normal_moments[bf, m] += wts[k] * nvals[bf, k] * linevals[m, k]
        print(normal_moments)
        print()
    print()
    for ed in range(3):
        t = T.compute_edge_tangent(ed)
        print(t)
        wts = np.asarray(Qline.wts)

        vals = MTW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        tvals = np.dot(np.transpose(vals, (0, 2, 1)), t)
        tangent_moments = np.zeros(9)
        for bf in range(9):
            for k in range(len(Qline.wts)):
                tangent_moments[bf] += wts[k] * tvals[bf, k] * linevals[0, k]
        print(tangent_moments)


if __name__ == "__main__":
    test_dofs()
