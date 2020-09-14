import numpy as np
from FIAT import ufc_simplex, ArnoldWintherNC, make_quadrature, expansions


def test_dofs():
    line = ufc_simplex(1)
    T = ufc_simplex(2)
    T.vertices = np.random.rand(3, 2)
    AW = ArnoldWintherNC(T, 2)

    Qline = make_quadrature(line, 6)

    linebfs = expansions.LineExpansionSet(line)
    linevals = linebfs.tabulate(1, Qline.pts)

    # n, n moments
    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        wts = np.asarray(Qline.wts)
        nqpline = len(wts)

        vals = AW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        nnvals = np.zeros((18, nqpline))
        for i in range(18):
            for j in range(len(wts)):
                nnvals[i, j] = n @ vals[i, :, :, j] @ n

        nnmoments = np.zeros((18, 2))

        for bf in range(18):
            for k in range(nqpline):
                for m in (0, 1):
                    nnmoments[bf, m] += wts[k] * nnvals[bf, k] * linevals[m, k]

        for bf in range(18):
            if bf != AW.dual.entity_ids[1][ed][0] and bf != AW.dual.entity_ids[1][ed][2]:
                assert np.allclose(nnmoments[bf, :], np.zeros(2))

    # n, t moments
    for ed in range(3):
        n = T.compute_scaled_normal(ed)
        t = T.compute_edge_tangent(ed)
        wts = np.asarray(Qline.wts)
        nqpline = len(wts)

        vals = AW.tabulate(0, Qline.pts, (1, ed))[(0, 0)]
        ntvals = np.zeros((18, nqpline))
        for i in range(18):
            for j in range(len(wts)):
                ntvals[i, j] = n @ vals[i, :, :, j] @ t

        ntmoments = np.zeros((18, 2))

        for bf in range(18):
            for k in range(nqpline):
                for m in (0, 1):
                    ntmoments[bf, m] += wts[k] * ntvals[bf, k] * linevals[m, k]

        for bf in range(18):
            if bf != AW.dual.entity_ids[1][ed][1] and bf != AW.dual.entity_ids[1][ed][3]:
                assert np.allclose(ntmoments[bf, :], np.zeros(2), atol=1.e-7)

    # check internal dofs
    Q = make_quadrature(T, 6)
    qpvals = AW.tabulate(0, Q.pts)[(0, 0)]
    const_moms = qpvals @ Q.wts
    assert np.allclose(const_moms[:12], np.zeros((12, 2, 2)))
    assert np.allclose(const_moms[15:], np.zeros((3, 2, 2)))
    assert np.allclose(const_moms[12:15, 0, 0], np.asarray([1, 0, 0]))
    assert np.allclose(const_moms[12:15, 0, 1], np.asarray([0, 1, 0]))
    assert np.allclose(const_moms[12:15, 1, 0], np.asarray([0, 1, 0]))
    assert np.allclose(const_moms[12:15, 1, 1], np.asarray([0, 0, 1]))
