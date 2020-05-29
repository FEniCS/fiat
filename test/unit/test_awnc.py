import numpy as np


def test_dofs():
    from FIAT import ufc_simplex, ArnoldWintherNC, make_quadrature, expansions

    line = ufc_simplex(1)
    T = ufc_simplex(2)
    T.vertices = np.random.rand(3, 2)
    AW = ArnoldWintherNC(T, 2)

    Qline = make_quadrature(line, 6)

    linebfs = expansions.LineExpansionSet(line)
    linevals = linebfs.tabulate(1, Qline.pts)

    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        wts = np.asarray(Qline.wts)
        nqpline = len(wts)

        vals = AW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        nnvals = np.zeros((18, nqpline))
        for i in range(18):
            for j in range(len(wts)):
                nnvals[i, j] = np.dot(n, vals[i, :, :, j] @ n)

        nnmoments = np.zeros((18, 2))

        for bf in range(18):
            for k in range(nqpline):
                for m in (0, 1):
                    nnmoments[bf, m] += wts[k] * nnvals[bf, k] * linevals[m, k]
        print(nnmoments)
        print()
    print()
    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        t = T.compute_edge_tangent(ed)
        wts = np.asarray(Qline.wts)
        nqpline = len(wts)

        vals = AW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        ntvals = np.zeros((18, nqpline))
        for i in range(18):
            for j in range(len(wts)):
                ntvals[i, j] = np.dot(n, vals[i, :, :, j] @ t)

        ntmoments = np.zeros((18, 2))

        for bf in range(18):
            for k in range(nqpline):
                for m in (0, 1):
                    ntmoments[bf, m] += wts[k] * ntvals[bf, k] * linevals[m, k]
        print(ntmoments)
        print()



if __name__ == "__main__":
    test_dofs()
