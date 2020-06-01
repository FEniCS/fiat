import numpy as np


def test_dofs():
    from FIAT import ufc_simplex, ArnoldWinther, make_quadrature, expansions

    line = ufc_simplex(1)
    T = ufc_simplex(2)
    T.vertices = np.random.rand(3, 2)
    AW = ArnoldWinther(T, 3)

    # check Kronecker property at vertices

    bases = [[[1, 0], [0, 0]], [[0, 1], [1, 0]], [[0, 0], [0, 1]]]

    vert_vals = AW.tabulate(0, T.vertices)[(0, 0)]
    for i in range(3):
        for j in range(3):
            assert np.allclose(vert_vals[3*i+j, :, :, i], bases[j])
            for k in (1, 2):
                assert np.allclose(vert_vals[3*i+j, :, :, (i+k) % 3], np.zeros((2, 2)))

    # check edge moments
    Qline = make_quadrature(line, 6)

    linebfs = expansions.LineExpansionSet(line)
    linevals = linebfs.tabulate(1, Qline.pts)

    # n, n moments
    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        wts = np.asarray(Qline.wts)
        nqpline = len(wts)

        vals = AW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        nnvals = np.zeros((30, nqpline))
        for i in range(30):
            for j in range(len(wts)):
                nnvals[i, j] = n @ vals[i, :, :, j] @ n

        nnmoments = np.zeros((30, 2))

        for bf in range(30):
            for k in range(nqpline):
                for m in (0, 1):
                    nnmoments[bf, m] += wts[k] * nnvals[bf, k] * linevals[m, k]

        for bf in range(30):
            if bf != AW.dual.entity_ids[1][ed][0] and bf != AW.dual.entity_ids[1][ed][2]:
                assert np.allclose(nnmoments[bf, :], np.zeros(2))

    # n, t moments
    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        t = T.compute_edge_tangent(ed)
        wts = np.asarray(Qline.wts)
        nqpline = len(wts)

        vals = AW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        ntvals = np.zeros((30, nqpline))
        for i in range(30):
            for j in range(len(wts)):
                ntvals[i, j] = n @ vals[i, :, :, j] @ t

        ntmoments = np.zeros((30, 2))

        for bf in range(30):
            for k in range(nqpline):
                for m in (0, 1):
                    ntmoments[bf, m] += wts[k] * ntvals[bf, k] * linevals[m, k]

        for bf in range(30):
            if bf != AW.dual.entity_ids[1][ed][1] and bf != AW.dual.entity_ids[1][ed][3]:
                assert np.allclose(ntmoments[bf, :], np.zeros(2))

    # check internal dofs
    Q = make_quadrature(T, 6)
    qpvals = AW.tabulate(0, Q.pts)[(0, 0)]
    const_moms = qpvals @ Q.wts
    assert np.allclose(const_moms[:21], np.zeros((21, 2, 2)))
    assert np.allclose(const_moms[24:], np.zeros((6, 2, 2)))
    assert np.allclose(const_moms[21:24, 0, 0], np.asarray([1, 0, 0]))
    assert np.allclose(const_moms[21:24, 0, 1], np.asarray([0, 1, 0]))
    assert np.allclose(const_moms[21:24, 1, 0], np.asarray([0, 1, 0]))
    assert np.allclose(const_moms[21:24, 1, 1], np.asarray([0, 0, 1]))


def frob(a, b):
    return a.ravel() @ b.ravel()


def test_projection():
    from FIAT import ufc_simplex, ArnoldWinther, make_quadrature, expansions

    T = ufc_simplex(2)
    # T.vertices = np.random.rand(3, 2)
    AW = ArnoldWinther(T, 3)

    Q = make_quadrature(T, 3)
    qpts = Q.pts
    qwts = Q.wts
    nqp = len(Q.wts)

    nbf = 24
    m = np.zeros((24, 24))
    # b = np.zeros((24,))

    bfvals = AW.tabulate(0, qpts)[(0, 0)]

    for i in range(nbf):
        for j in range(nbf):
            for k in range(nqp):
                for ell in range(2):
                    for em in range(2):
                        m[i, j] += qwts[k] * frob(bfvals[i, :, :, k],
                                                  bfvals[j, :, :, k])

    print(np.linalg.cond(m))
    print(np.linalg.svd(m, compute_uv=False))

if __name__ == "__main__":
    # test_dofs()
    test_projection()
