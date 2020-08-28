"""Quadrature schemes on cells

This module generates quadrature schemes on reference cells that integrate
exactly a polynomial of a given degree using a specified scheme.

Scheme options are:

  scheme="default"

  scheme="canonical" (collapsed Gauss scheme)

Background on the schemes:

  Keast rules for tetrahedra:
    Keast, P. Moderate-degree tetrahedral quadrature formulas, Computer
    Methods in Applied Mechanics and Engineering 55(3):339-348, 1986.
    http://dx.doi.org/10.1016/0045-7825(86)90059-9
"""

# Copyright (C) 2011 Garth N. Wells
# Copyright (C) 2016 Miklos Homolya
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# First added:  2011-04-19
# Last changed: 2011-04-19

# NumPy
from numpy import array, arange, float64

# FIAT
from FIAT.reference_element import QUADRILATERAL, HEXAHEDRON, TENSORPRODUCT, UFCTriangle, UFCTetrahedron
from FIAT.quadrature import QuadratureRule, make_quadrature, make_tensor_product_quadrature


def create_quadrature(ref_el, degree, scheme="default"):
    """
    Generate quadrature rule for given reference element
    that will integrate an polynomial of order 'degree' exactly.

    For low-degree (<=6) polynomials on triangles and tetrahedra, this
    uses hard-coded rules, otherwise it falls back to a collapsed
    Gauss scheme on simplices.  On tensor-product cells, it is a
    tensor-product quadrature rule of the subcells.

    :arg cell: The FIAT cell to create the quadrature for.
    :arg degree: The degree of polynomial that the rule should
        integrate exactly.
    """
    if ref_el.get_shape() == TENSORPRODUCT:
        try:
            degree = tuple(degree)
        except TypeError:
            degree = (degree,) * len(ref_el.cells)

        assert len(ref_el.cells) == len(degree)
        quad_rules = [create_quadrature(c, d, scheme)
                      for c, d in zip(ref_el.cells, degree)]
        return make_tensor_product_quadrature(*quad_rules)

    if ref_el.get_shape() in [QUADRILATERAL, HEXAHEDRON]:
        return create_quadrature(ref_el.product, degree, scheme)

    if degree < 0:
        raise ValueError("Need positive degree, not %d" % degree)

    if scheme == "default":
        # TODO: Point transformation to support good schemes on
        # non-UFC reference elements.
        if isinstance(ref_el, UFCTriangle):
            return _triangle_scheme(degree)
        elif isinstance(ref_el, UFCTetrahedron):
            return _tetrahedron_scheme(degree)
        else:
            return _fiat_scheme(ref_el, degree)
    elif scheme == "canonical":
        return _fiat_scheme(ref_el, degree)
    elif scheme == "KMV":  # Kong-Mulder-Veldhuizen scheme
        return _kmv_lump_scheme(ref_el, degree)
    else:
        raise ValueError("Unknown quadrature scheme: %s." % scheme)


def _fiat_scheme(ref_el, degree):
    """Get quadrature scheme from FIAT interface"""

    # Number of points per axis for exact integration
    num_points_per_axis = (degree + 1 + 1) // 2

    # Create and return FIAT quadrature rule
    return make_quadrature(ref_el, num_points_per_axis)


def _kmv_lump_scheme(ref_el, degree):
    """Specialized quadrature schemes for P < 6 for KMV simplical elements."""

    sd = ref_el.get_spatial_dimension()
    # set the unit element
    if sd == 2:
        T = UFCTriangle()
    elif sd == 3:
        T = UFCTetrahedron()
    else:
        raise ValueError("Dimension not supported")

    if degree == 1:
        x = ref_el.vertices
        w = arange(sd + 1, dtype=float64)
        if sd == 2:
            w[:] = 1.0 / 6.0
        elif sd == 3:
            w[:] = 1.0 / 24.0
        else:
            raise ValueError("Dimension not supported")
    elif degree == 2:
        if sd == 2:
            x = list(ref_el.vertices)
            for e in range(3):
                x.extend(ref_el.make_points(1, e, 2))  # edge midpoints
            x.extend(ref_el.make_points(2, 0, 3))  # barycenter
            w = arange(7, dtype=float64)
            w[0:3] = 1.0 / 40.0
            w[3:6] = 1.0 / 15.0
            w[6] = 9.0 / 40.0
        elif sd == 3:
            x = list(ref_el.vertices)
            x.extend(
                [
                    (0.0, 0.50, 0.50),
                    (0.50, 0.0, 0.50),
                    (0.50, 0.50, 0.0),
                    (0.0, 0.0, 0.50),
                    (0.0, 0.50, 0.0),
                    (0.50, 0.0, 0.0),
                ]
            )
            # in facets
            x.extend(
                [
                    (0.33333333333333337, 0.3333333333333333, 0.3333333333333333),
                    (0.0, 0.3333333333333333, 0.3333333333333333),
                    (0.3333333333333333, 0.0, 0.3333333333333333),
                    (0.3333333333333333, 0.3333333333333333, 0.0),
                ]
            )
            # in the cell
            x.extend([(1 / 4, 1 / 4, 1 / 4)])
            w = arange(15, dtype=float64)
            w[0:4] = 17.0 / 5040.0
            w[4:10] = 2.0 / 315.0
            w[10:14] = 9.0 / 560.0
            w[14] = 16.0 / 315.0
        else:
            raise ValueError("Dimension not supported")

    elif degree == 3:
        if sd == 2:
            alpha = 0.2934695559090401
            beta = 0.2073451756635909
            x = list(ref_el.vertices)
            x.extend(
                [
                    (1 - alpha, alpha),
                    (alpha, 1 - alpha),
                    (0.0, 1 - alpha),
                    (0.0, alpha),
                    (alpha, 0.0),
                    (1 - alpha, 0.0),
                ]  # edge points
            )
            x.extend(
                [(beta, beta), (1 - 2 * beta, beta), (beta, 1 - 2 * beta)]
            )  # points in center of cell
            w = arange(12, dtype=float64)
            w[0:3] = 0.007436456512410291
            w[3:9] = 0.02442084061702551
            w[9:12] = 0.1103885289202054
        elif sd == 3:
            x = list(ref_el.vertices)
            x.extend(
                [
                    (0, 0.685789657581967, 0.314210342418033),
                    (0, 0.314210342418033, 0.685789657581967),
                    (0.314210342418033, 0, 0.685789657581967),
                    (0.685789657581967, 0, 0.314210342418033),
                    (0.685789657581967, 0.314210342418033, 0.0),
                    (0.314210342418033, 0.685789657581967, 0.0),
                    (0, 0, 0.685789657581967),
                    (0, 0, 0.314210342418033),
                    (0, 0.314210342418033, 0.0),
                    (0, 0.685789657581967, 0.0),
                    (0.314210342418033, 0, 0.0),
                    (0.685789657581967, 0, 0.0),
                ]
            )  # 12 points on edges of facets (0-->1-->2)

            x.extend(
                [
                    (0.21548220313557542, 0.5690355937288492, 0.21548220313557542),
                    (0.21548220313557542, 0.21548220313557542, 0.5690355937288492),
                    (0.5690355937288492, 0.21548220313557542, 0.21548220313557542),
                    (0.0, 0.5690355937288492, 0.21548220313557542),
                    (0.0, 0.21548220313557542, 0.5690355937288492),
                    (0.0, 0.21548220313557542, 0.21548220313557542),
                    (0.5690355937288492, 0.0, 0.21548220313557542),
                    (0.21548220313557542, 0.0, 0.5690355937288492),
                    (0.21548220313557542, 0.0, 0.21548220313557542),
                    (0.5690355937288492, 0.21548220313557542, 0.0),
                    (0.21548220313557542, 0.5690355937288492, 0.0),
                    (0.21548220313557542, 0.21548220313557542, 0.0),
                ]
            )  # 12 points (3 points on each facet, 1st two parallel to edge 0)
            alpha = 1 / 6
            x.extend(
                [
                    (alpha, alpha, 0.5),
                    (0.5, alpha, alpha),
                    (alpha, 0.5, alpha),
                    (alpha, alpha, alpha),
                ]
            )  # 4 points inside the cell
            w = arange(32, dtype=float64)
            w[0:4] = 0.00068688236002531922325120561367839
            w[4:16] = 0.0015107814913526136472998739890272
            w[16:28] = 0.0050062894680040258624242888174649
            w[28:32] = 0.021428571428571428571428571428571
        else:
            raise ValueError("Dimension not supported")
    elif degree == 4:
        if sd == 2:
            alpha = 0.2113248654051871  # 0.2113248654051871
            beta1 = 0.4247639617258106  # 0.4247639617258106
            beta2 = 0.130791593829745  # 0.130791593829745
            x = list(ref_el.vertices)
            for e in range(3):
                x.extend(ref_el.make_points(1, e, 2))  # edge midpoints
            x.extend(
                [
                    (1 - alpha, alpha),
                    (alpha, 1 - alpha),
                    (0.0, 1 - alpha),
                    (0.0, alpha),
                    (alpha, 0.0),
                    (1 - alpha, 0.0),
                ]  # edge points
            )
            x.extend(
                [(beta1, beta1), (1 - 2 * beta1, beta1), (beta1, 1 - 2 * beta1)]
            )  # points in center of cell
            x.extend(
                [(beta2, beta2), (1 - 2 * beta2, beta2), (beta2, 1 - 2 * beta2)]
            )  # points in center of cell
            w = arange(18, dtype=float64)
            w[0:3] = 0.003174603174603175  # chk
            w[3:6] = 0.0126984126984127  # chk 0.0126984126984127
            w[6:12] = 0.01071428571428571  # chk 0.01071428571428571
            w[12:15] = 0.07878121446939182  # chk 0.07878121446939182
            w[15:18] = 0.05058386489568756  # chk 0.05058386489568756
        else:
            raise ValueError("Dimension not supported")

    elif degree == 5:
        if sd == 2:
            alpha1 = 0.3632980741536860e-00
            alpha2 = 0.1322645816327140e-00
            beta1 = 0.4578368380791611e-00
            beta2 = 0.2568591072619591e-00
            beta3 = 0.5752768441141011e-01
            gamma1 = 0.7819258362551702e-01
            delta1 = 0.2210012187598900e-00
            x = list(ref_el.vertices)
            x.extend(
                [
                    (1 - alpha1, alpha1),
                    (alpha1, 1 - alpha1),
                    (0.0, 1 - alpha1),
                    (0.0, alpha1),
                    (alpha1, 0.0),
                    (1 - alpha1, 0.0),
                ]  # edge points
            )
            x.extend(
                [
                    (1 - alpha2, alpha2),
                    (alpha2, 1 - alpha2),
                    (0.0, 1 - alpha2),
                    (0.0, alpha2),
                    (alpha2, 0.0),
                    (1 - alpha2, 0.0),
                ]  # edge points
            )
            x.extend(
                [(beta1, beta1), (1 - 2 * beta1, beta1), (beta1, 1 - 2 * beta1)]
            )  # points in center of cell
            x.extend(
                [(beta2, beta2), (1 - 2 * beta2, beta2), (beta2, 1 - 2 * beta2)]
            )  # points in center of cell
            x.extend(
                [(beta3, beta3), (1 - 2 * beta3, beta3), (beta3, 1 - 2 * beta3)]
            )  # points in center of cell
            x.extend(
                [
                    (gamma1, delta1),
                    (1 - gamma1 - delta1, delta1),
                    (gamma1, 1 - gamma1 - delta1),
                    (delta1, gamma1),
                    (1 - gamma1 - delta1, gamma1),
                    (delta1, 1 - gamma1 - delta1),
                ]  # edge points
            )
            w = arange(30, dtype=float64)
            w[0:3] = 0.7094239706792450e-03
            w[3:9] = 0.6190565003676629e-02
            w[9:15] = 0.3480578640489211e-02
            w[15:18] = 0.3453043037728279e-01
            w[18:21] = 0.4590123763076286e-01
            w[21:24] = 0.1162613545961757e-01
            w[24:30] = 0.2727857596999626e-01
        else:
            raise ValueError("Dimension not supported")

    # Return scheme
    return QuadratureRule(T, x, w)


def _triangle_scheme(degree):
    """Return a quadrature scheme on a triangle of specified order. Falls
    back on canonical rule for higher orders."""

    if degree == 0 or degree == 1:
        # Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
        x = array([[1.0/3.0, 1.0/3.0]])
        w = array([0.5])
    elif degree == 2:
        # Scheme from Strang and Fix, 3 points, degree of precision 2
        x = array([[1.0/6.0, 1.0/6.0],
                   [1.0/6.0, 2.0/3.0],
                   [2.0/3.0, 1.0/6.0]])
        w = arange(3, dtype=float64)
        w[:] = 1.0/6.0
    elif degree == 3:
        # Scheme from Strang and Fix, 6 points, degree of precision 3
        x = array([[0.659027622374092, 0.231933368553031],
                   [0.659027622374092, 0.109039009072877],
                   [0.231933368553031, 0.659027622374092],
                   [0.231933368553031, 0.109039009072877],
                   [0.109039009072877, 0.659027622374092],
                   [0.109039009072877, 0.231933368553031]])
        w = arange(6, dtype=float64)
        w[:] = 1.0/12.0
    elif degree == 4:
        # Scheme from Strang and Fix, 6 points, degree of precision 4
        x = array([[0.816847572980459, 0.091576213509771],
                   [0.091576213509771, 0.816847572980459],
                   [0.091576213509771, 0.091576213509771],
                   [0.108103018168070, 0.445948490915965],
                   [0.445948490915965, 0.108103018168070],
                   [0.445948490915965, 0.445948490915965]])
        w = arange(6, dtype=float64)
        w[0:3] = 0.109951743655322
        w[3:6] = 0.223381589678011
        w = w/2.0
    elif degree == 5:
        # Scheme from Strang and Fix, 7 points, degree of precision 5
        x = array([[0.33333333333333333, 0.33333333333333333],
                   [0.79742698535308720, 0.10128650732345633],
                   [0.10128650732345633, 0.79742698535308720],
                   [0.10128650732345633, 0.10128650732345633],
                   [0.05971587178976981, 0.47014206410511505],
                   [0.47014206410511505, 0.05971587178976981],
                   [0.47014206410511505, 0.47014206410511505]])
        w = arange(7, dtype=float64)
        w[0] = 0.22500000000000000
        w[1:4] = 0.12593918054482717
        w[4:7] = 0.13239415278850616
        w = w/2.0
    elif degree == 6:
        # Scheme from Strang and Fix, 12 points, degree of precision 6
        x = array([[0.873821971016996, 0.063089014491502],
                   [0.063089014491502, 0.873821971016996],
                   [0.063089014491502, 0.063089014491502],
                   [0.501426509658179, 0.249286745170910],
                   [0.249286745170910, 0.501426509658179],
                   [0.249286745170910, 0.249286745170910],
                   [0.636502499121399, 0.310352451033785],
                   [0.636502499121399, 0.053145049844816],
                   [0.310352451033785, 0.636502499121399],
                   [0.310352451033785, 0.053145049844816],
                   [0.053145049844816, 0.636502499121399],
                   [0.053145049844816, 0.310352451033785]])
        w = arange(12, dtype=float64)
        w[0:3] = 0.050844906370207
        w[3:6] = 0.116786275726379
        w[6:12] = 0.082851075618374
        w = w/2.0
    else:
        # Get canonical scheme
        return _fiat_scheme(UFCTriangle(), degree)

    # Return scheme
    return QuadratureRule(UFCTriangle(), x, w)


def _tetrahedron_scheme(degree):
    """Return a quadrature scheme on a tetrahedron of specified
    degree. Falls back on canonical rule for higher orders"""

    if degree == 0 or degree == 1:
        # Scheme from Zienkiewicz and Taylor, 1 point, degree of precision 1
        x = array([[1.0/4.0, 1.0/4.0, 1.0/4.0]])
        w = array([1.0/6.0])
    elif degree == 2:
        # Scheme from Zienkiewicz and Taylor, 4 points, degree of precision 2
        a, b = 0.585410196624969, 0.138196601125011
        x = array([[a, b, b],
                   [b, a, b],
                   [b, b, a],
                   [b, b, b]])
        w = arange(4, dtype=float64)
        w[:] = 1.0/24.0
    elif degree == 3:
        # Scheme from Zienkiewicz and Taylor, 5 points, degree of precision 3
        # Note: this scheme has a negative weight
        x = array([[0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
                   [0.5000000000000000, 0.1666666666666666, 0.1666666666666666],
                   [0.1666666666666666, 0.5000000000000000, 0.1666666666666666],
                   [0.1666666666666666, 0.1666666666666666, 0.5000000000000000],
                   [0.1666666666666666, 0.1666666666666666, 0.1666666666666666]])
        w = arange(5, dtype=float64)
        w[0] = -0.8
        w[1:5] = 0.45
        w = w/6.0
    elif degree == 4:
        # Keast rule, 14 points, degree of precision 4
        # Values taken from http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
        # (KEAST5)
        x = array([[0.0000000000000000, 0.5000000000000000, 0.5000000000000000],
                   [0.5000000000000000, 0.0000000000000000, 0.5000000000000000],
                   [0.5000000000000000, 0.5000000000000000, 0.0000000000000000],
                   [0.5000000000000000, 0.0000000000000000, 0.0000000000000000],
                   [0.0000000000000000, 0.5000000000000000, 0.0000000000000000],
                   [0.0000000000000000, 0.0000000000000000, 0.5000000000000000],
                   [0.6984197043243866, 0.1005267652252045, 0.1005267652252045],
                   [0.1005267652252045, 0.1005267652252045, 0.1005267652252045],
                   [0.1005267652252045, 0.1005267652252045, 0.6984197043243866],
                   [0.1005267652252045, 0.6984197043243866, 0.1005267652252045],
                   [0.0568813795204234, 0.3143728734931922, 0.3143728734931922],
                   [0.3143728734931922, 0.3143728734931922, 0.3143728734931922],
                   [0.3143728734931922, 0.3143728734931922, 0.0568813795204234],
                   [0.3143728734931922, 0.0568813795204234, 0.3143728734931922]])
        w = arange(14, dtype=float64)
        w[0:6] = 0.0190476190476190
        w[6:10] = 0.0885898247429807
        w[10:14] = 0.1328387466855907
        w = w/6.0
    elif degree == 5:
        # Keast rule, 15 points, degree of precision 5
        # Values taken from http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
        # (KEAST6)
        x = array([[0.2500000000000000, 0.2500000000000000, 0.2500000000000000],
                   [0.0000000000000000, 0.3333333333333333, 0.3333333333333333],
                   [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                   [0.3333333333333333, 0.3333333333333333, 0.0000000000000000],
                   [0.3333333333333333, 0.0000000000000000, 0.3333333333333333],
                   [0.7272727272727273, 0.0909090909090909, 0.0909090909090909],
                   [0.0909090909090909, 0.0909090909090909, 0.0909090909090909],
                   [0.0909090909090909, 0.0909090909090909, 0.7272727272727273],
                   [0.0909090909090909, 0.7272727272727273, 0.0909090909090909],
                   [0.4334498464263357, 0.0665501535736643, 0.0665501535736643],
                   [0.0665501535736643, 0.4334498464263357, 0.0665501535736643],
                   [0.0665501535736643, 0.0665501535736643, 0.4334498464263357],
                   [0.0665501535736643, 0.4334498464263357, 0.4334498464263357],
                   [0.4334498464263357, 0.0665501535736643, 0.4334498464263357],
                   [0.4334498464263357, 0.4334498464263357, 0.0665501535736643]])
        w = arange(15, dtype=float64)
        w[0] = 0.1817020685825351
        w[1:5] = 0.0361607142857143
        w[5:9] = 0.0698714945161738
        w[9:15] = 0.0656948493683187
        w = w/6.0
    elif degree == 6:
        # Keast rule, 24 points, degree of precision 6
        # Values taken from http://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
        # (KEAST7)
        x = array([[0.3561913862225449, 0.2146028712591517, 0.2146028712591517],
                   [0.2146028712591517, 0.2146028712591517, 0.2146028712591517],
                   [0.2146028712591517, 0.2146028712591517, 0.3561913862225449],
                   [0.2146028712591517, 0.3561913862225449, 0.2146028712591517],
                   [0.8779781243961660, 0.0406739585346113, 0.0406739585346113],
                   [0.0406739585346113, 0.0406739585346113, 0.0406739585346113],
                   [0.0406739585346113, 0.0406739585346113, 0.8779781243961660],
                   [0.0406739585346113, 0.8779781243961660, 0.0406739585346113],
                   [0.0329863295731731, 0.3223378901422757, 0.3223378901422757],
                   [0.3223378901422757, 0.3223378901422757, 0.3223378901422757],
                   [0.3223378901422757, 0.3223378901422757, 0.0329863295731731],
                   [0.3223378901422757, 0.0329863295731731, 0.3223378901422757],
                   [0.2696723314583159, 0.0636610018750175, 0.0636610018750175],
                   [0.0636610018750175, 0.2696723314583159, 0.0636610018750175],
                   [0.0636610018750175, 0.0636610018750175, 0.2696723314583159],
                   [0.6030056647916491, 0.0636610018750175, 0.0636610018750175],
                   [0.0636610018750175, 0.6030056647916491, 0.0636610018750175],
                   [0.0636610018750175, 0.0636610018750175, 0.6030056647916491],
                   [0.0636610018750175, 0.2696723314583159, 0.6030056647916491],
                   [0.2696723314583159, 0.6030056647916491, 0.0636610018750175],
                   [0.6030056647916491, 0.0636610018750175, 0.2696723314583159],
                   [0.0636610018750175, 0.6030056647916491, 0.2696723314583159],
                   [0.2696723314583159, 0.0636610018750175, 0.6030056647916491],
                   [0.6030056647916491, 0.2696723314583159, 0.0636610018750175]])
        w = arange(24, dtype=float64)
        w[0:4] = 0.0399227502581679
        w[4:8] = 0.0100772110553207
        w[8:12] = 0.0553571815436544
        w[12:24] = 0.0482142857142857
        w = w/6.0
    else:
        # Get canonical scheme
        return _fiat_scheme(UFCTetrahedron(), degree)

    # Return scheme
    return QuadratureRule(UFCTetrahedron(), x, w)
