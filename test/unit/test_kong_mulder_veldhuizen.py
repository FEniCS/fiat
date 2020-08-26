import numpy as np
import pytest

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT import create_quadrature, make_quadrature, polynomial_set
from FIAT.kong_mulder_veldhuizen import KongMulderVeldhuizen as KMV

I = UFCInterval()
T = UFCTriangle()
Te = UFCTetrahedron()


@pytest.mark.parametrize("p_d", [(1, 1), (2, 3), (3, 4)])
def test_kmv_quad_tet_schemes(p_d):  # noqa: W503
    fct = np.math.factorial
    p, d = p_d
    q = create_quadrature(Te, p, "KMV")
    for i in range(d + 1):
        for j in range(d + 1 - i):
            for k in range(d + 1 - i - j):
                trueval = fct(i) * fct(j) * fct(k) / fct(i + j + k + 3)
                assert (
                    np.abs(
                        trueval -
                        q.integrate(lambda x: x[0] ** i * x[1] ** j * x[2] ** k)
                    ) <
                    1.0e-10
                )


@pytest.mark.parametrize("p_d", [(1, 1), (2, 3), (3, 5), (4, 7), (5, 9)])
def test_kmv_quad_tri_schemes(p_d):
    fct = np.math.factorial
    p, d = p_d
    q = create_quadrature(T, p, "KMV")
    for i in range(d + 1):
        for j in range(d + 1 - i):
            trueval = fct(i) * fct(j) / fct(i + j + 2)
            assert (
                np.abs(trueval - q.integrate(lambda x: x[0] ** i * x[1] ** j)) < 1.0e-10
            )


@pytest.mark.parametrize(
    "element_degree",
    [(KMV(T, 1), 1), (KMV(T, 2), 2), (KMV(T, 3), 3), (KMV(T, 4), 4), (KMV(T, 5), 5)],
)
def test_Kronecker_property_tris(element_degree):
    """
    Evaluating the nodal basis at the special quadrature points should
    have a Kronecker property.  Also checks that the basis functions
    and quadrature points are given the same ordering.
    """
    element, degree = element_degree
    qr = create_quadrature(T, degree, scheme="KMV")
    (basis,) = element.tabulate(0, qr.get_points()).values()
    assert np.allclose(basis, np.eye(*basis.shape))


@pytest.mark.parametrize(
    "element_degree", [(KMV(Te, 1), 1), (KMV(Te, 2), 2), (KMV(Te, 3), 3)]
)
def test_Kronecker_property_tets(element_degree):
    """
    Evaluating the nodal basis at the special quadrature points should
    have a Kronecker property.  Also checks that the basis functions
    and quadrature points are given the same ordering.
    """
    element, degree = element_degree
    qr = create_quadrature(Te, degree, scheme="KMV")
    (basis,) = element.tabulate(0, qr.get_points()).values()
    assert np.allclose(basis, np.eye(*basis.shape))


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_edge_degree(degree):
    """Verify that the outer edges of a degree KMV element
       are indeed of degree and the interior is of degree+1"""
    # create a degree+1 polynomial
    I = UFCInterval()
    # an exact quad. rule for a degree+1 polynomial on the UFCinterval
    qr = make_quadrature(I, degree + 1)
    W = np.diag(qr.wts)
    sd = I.get_spatial_dimension()
    pset = polynomial_set.ONPolynomialSet(I, degree + 1, (sd,))
    pset = pset.take([degree + 1])
    # tabulate at the quadrature points
    interval_vals = pset.tabulate(qr.get_points())[(0,)]
    interval_vals = np.squeeze(interval_vals)
    # create degree KMV element (should have degree outer edges and degree+1 edge in center)
    T = UFCTriangle()
    element = KMV(T, degree)
    # tabulate values on an edge of the KMV element
    for e in range(3):
        edge_values = element.tabulate(0, qr.get_points(), (1, e))[(0, 0)]
        # degree edge should be orthogonal to degree+1 ONpoly edge values
        result = edge_values @ W @ interval_vals.T
        assert np.allclose(np.sum(result), 0.0)


@pytest.mark.parametrize(
    "element_degree",
    [(KMV(T, 1), 1), (KMV(T, 2), 2), (KMV(T, 3), 3), (KMV(T, 4), 4), (KMV(T, 5), 5)],
)
def test_interpolate_monomials_tris(element_degree):
    element, degree = element_degree

    # ordered the same way as KMV nodes
    pts = create_quadrature(T, degree, "KMV").pts

    Q = make_quadrature(T, 2 * degree)
    phis = element.tabulate(0, Q.pts)[0, 0]
    print("deg", degree)
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            m = lambda x: x[0] ** i * x[1] ** j
            dofs = np.array([m(pt) for pt in pts])
            interp = phis.T @ dofs
            matqp = np.array([m(pt) for pt in Q.pts])
            err = 0.0
            for k in range(phis.shape[1]):
                err += Q.wts[k] * (interp[k] - matqp[k]) ** 2
            assert np.sqrt(err) <= 1.0e-12


@pytest.mark.parametrize(
    "element_degree", [(KMV(Te, 1), 1), (KMV(Te, 2), 2), (KMV(Te, 3), 3)]
)
def test_interpolate_monomials_tets(element_degree):
    element, degree = element_degree

    # ordered the same way as KMV nodes
    pts = create_quadrature(Te, degree, "KMV").pts

    Q = make_quadrature(Te, 2 * degree)
    phis = element.tabulate(0, Q.pts)[0, 0, 0]
    print("deg", degree)
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            for k in range(degree + 1 - i - j):
                m = lambda x: x[0] ** i * x[1] ** j * x[2] ** k
                dofs = np.array([m(pt) for pt in pts])
                interp = phis.T @ dofs
                matqp = np.array([m(pt) for pt in Q.pts])
                err = 0.0
                for kk in range(phis.shape[1]):
                    err += Q.wts[kk] * (interp[kk] - matqp[kk]) ** 2
                assert np.sqrt(err) <= 1.0e-12
