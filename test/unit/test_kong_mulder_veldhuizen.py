import numpy as np
import pytest

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT import create_quadrature, make_quadrature, polynomial_set
from FIAT.kong_mulder_veldhuizen import KongMulderVeldhuizen as KMV

I = UFCInterval()
T = UFCTriangle()
Te = UFCTetrahedron()


@pytest.mark.parametrize(
    "p_d_t", [(1, 1, T), (2, 3, T), (3, 5, T), (4, 7, T), (5, 9, T), (1, 1, Te)]
)
def test_kmv_quad_schemes(p_d_t):
    fct = np.math.factorial
    p, d, t = p_d_t
    q = create_quadrature(t, p, "KMV")
    for i in range(d + 1):
        for j in range(d + 1 - i):
            trueval = fct(i) * fct(j) / fct(i + j + 2)
            assert (
                np.abs(trueval - q.integrate(lambda x: x[0] ** i * x[1] ** j)) < 1.0e-10
            )


@pytest.mark.parametrize(
    "element_degree",
    [(KMV(T, 1), 1), (KMV(T, 2), 2), (KMV(T, 3), 3), (KMV(T, 4), 4), (KMV(T, 5), 5)]
)
def test_Kronecker_property(element_degree):
    """
    Evaluating the nodal basis at the special quadrature points should
    have a Kronecker property.  Also checks that the basis functions
    and quadrature points are given the same ordering.
    """
    element, degree = element_degree
    qr = create_quadrature(T, degree, scheme="KMV")
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
def test_interpolate_monomials(element_degree):
    element, degree = element_degree
    T = UFCTriangle()

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
