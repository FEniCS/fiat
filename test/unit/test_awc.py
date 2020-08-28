import numpy as np
from FIAT import ufc_simplex, ArnoldWinther, make_quadrature, expansions


def test_dofs():
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
    T = ufc_simplex(2)
    T.vertices = np.asarray([(0.0, 0.0), (1.0, 0.0), (0.5, 2.1)])

    AW = ArnoldWinther(T, 3)

    Q = make_quadrature(T, 4)
    qpts = np.asarray(Q.pts)
    qwts = np.asarray(Q.wts)
    nqp = len(Q.wts)

    nbf = 24
    m = np.zeros((nbf, nbf))
    b = np.zeros((24,))
    rhs_vals = np.zeros((2, 2, nqp))

    bfvals = AW.tabulate(0, qpts)[(0, 0)][:nbf, :, :, :]

    for i in range(nbf):
        for j in range(nbf):
            for k in range(nqp):
                m[i, j] += qwts[k] * frob(bfvals[i, :, :, k],
                                          bfvals[j, :, :, k])

    assert np.linalg.cond(m) < 1.e12

    comps = [(0, 0), (0, 1), (0, 0)]

    # loop over monomials up to degree 2
    for deg in range(3):
        for jj in range(deg+1):
            ii = deg-jj
            for comp in comps:
                b[:] = 0.0
                # set RHS (symmetrically) to be the monomial in
                # the proper component.
                rhs_vals[comp] = qpts[:, 0]**ii * qpts[:, 1]**jj
                rhs_vals[tuple(reversed(comp))] = rhs_vals[comp]
                for i in range(nbf):
                    for k in range(nqp):
                        b[i] += qwts[k] * frob(bfvals[i, :, :, k],
                                               rhs_vals[:, :, k])
                x = np.linalg.solve(m, b)

                sol_at_qpts = np.zeros(rhs_vals.shape)
                for i in range(nbf):
                    for k in range(nqp):
                        sol_at_qpts[:, :, k] += x[i] * bfvals[i, :, :, k]

                diff = sol_at_qpts - rhs_vals
                err = 0.0
                for k in range(nqp):
                    err += qwts[k] * frob(diff[:, :, k], diff[:, :, k])

                assert np.sqrt(err) < 1.e-12
