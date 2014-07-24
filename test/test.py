# Copyright (C) 2010 Anders Logg
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2010-01-31
# Last changed: 2014-06-30

import nose
import sys
import json
import numpy

from FIAT import supported_elements, make_quadrature, ufc_simplex, \
    newdubiner, expansions, reference_element, polynomial_set

# Parameters
tolerance = 1e-8


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # If numpy array, convert it to a list and store it in a dict.
        if isinstance(obj, numpy.ndarray):
            data = obj.tolist()
            return dict(__ndarray__=data,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    # If dict and have '__ndarray__' as a key, convert it back to ndarray.
    if isinstance(dct, dict) and '__ndarray__' in dct:
        return numpy.asarray(dct['__ndarray__']).reshape(dct['shape'])
    return dct


def test_polynomials():
    def create_data():
        ps = polynomial_set.ONPolynomialSet(
            ref_el=reference_element.DefaultTetrahedron(),
            degree=3
            )
        return ps.dmats

    # Try reading reference values
    filename = "reference-polynomials.json"
    try:
        reference = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
    except IOError:
        reference = create_data()
        # Store the data for the future
        json.dump(reference, open(filename, "w"), cls=NumpyEncoder)

    dmats = create_data()

    for dmat, reference_dmat in zip(dmats, reference):
        assert (abs(dmat - reference_dmat) < tolerance).all()
    return

def test_polynomials_1D():
    def create_data():
        ps = polynomial_set.ONPolynomialSet(
            ref_el=reference_element.DefaultLine(),
            degree=3
            )
        return ps.dmats

    # Try reading reference values
    filename = "reference-polynomials.json"
    try:
        reference = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
    except IOError:
        reference = create_data()
        # Store the data for the future
        json.dump(reference, open(filename, "w"), cls=NumpyEncoder)

    dmats = create_data()

    for dmat, reference_dmat in zip(dmats, reference):
        assert (abs(dmat - reference_dmat) < tolerance).all()
    return


def test_expansions():
    def create_data():
        E = reference_element.DefaultTriangle()
        k = 3
        pts = E.make_lattice(k)
        Phis = expansions.get_expansion_set(E)
        phis = Phis.tabulate(k, pts)
        dphis = Phis.tabulate_derivatives(k, pts)
        return phis, dphis

    # Try reading reference values
    filename = "reference-expansions.json"
    try:
        reference = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
    except IOError:
        reference = create_data()
        # Convert reference to list of int
        json.dump(reference, open(filename, "w"), cls=NumpyEncoder)

    table_phi, table_dphi = create_data()
    reference_table_phi, reference_table_dphi = reference

    # Test raw point data
    diff = numpy.array(table_phi) - numpy.array(reference_table_phi)
    assert (abs(diff) < tolerance).all()

    # Test derivative values
    for entry, reference_entry in zip(table_dphi, reference_table_dphi):
        for point, reference_point in zip(entry, reference_entry):
            value, gradient = point[0], point[1]
            reference_value, reference_gradient = \
                reference_point[0], reference_point[1]
            assert abs(value - reference_value) < tolerance
            diff = numpy.array(gradient) - numpy.array(reference_gradient)
            assert (abs(diff) < tolerance).all()
    return


def test_expansions_jet():
    def create_data():
        latticeK = 2
        n = 1
        order = 2
        E = reference_element.DefaultTetrahedron()
        pts = E.make_lattice(latticeK)
        F = expansions.TetrahedronExpansionSet(E)
        return F.tabulate_jet(n, pts, order)

    filename = "reference-expansions-jet.json"
    try:
        reference_jet = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
    except IOError:
        reference_jet = create_data()
        # Store the data for the future
        json.dump(reference_jet, open(filename, "w"), cls=NumpyEncoder)

    # Test jet data
    data = create_data()
    reference_data = reference_jet
    for datum, reference_datum in zip(data, reference_data):
        diff = numpy.array(datum) - numpy.array(reference_datum)
        assert (abs(diff) < tolerance).all()

    return


def test_newdubiner():
    def create_data():
        latticeK = 2
        D = 3
        pts = newdubiner.make_tetrahedron_lattice(latticeK, float)
        return newdubiner.tabulate_tetrahedron_derivatives(D, pts, float)

    # Try reading reference values
    filename = "reference-newdubiner.json"
    try:
        reference = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
    except IOError:
        reference = create_data()
        # Convert reference to list of int
        json.dump(reference, open(filename, "w"), cls=NumpyEncoder)

    # Actually perform the test
    table = create_data()

    for data, reference_data in zip(table, reference):
        for point, reference_point in zip(data, reference_data):
            for k in range(2):
                diff = numpy.array(point[k]) - numpy.array(reference_point[k])
                assert (abs(diff) < tolerance).all()
    return


def test_newdubiner_jet():
    def create_data():
        latticeK = 2
        D = 3
        n = 1
        order = 2
        pts = newdubiner.make_tetrahedron_lattice(latticeK, float)
        return newdubiner.tabulate_jet(D, n, pts, order, float)

    filename = "reference-newdubiner-jet.json"
    try:
        reference_jet = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
    except IOError:
        reference_jet = create_data()
        # Store the data for the future
        json.dump(reference_jet, open(filename, "w"), cls=NumpyEncoder)

    table_jet = create_data()
    for datum, reference_datum in zip(table_jet, reference_jet):
        for entry, reference_entry in zip(datum, reference_datum):
            for k in range(3):
                diff = numpy.array(entry[k]) - numpy.array(reference_entry[k])
                assert (abs(diff) < tolerance).all()

    return


def test_quadrature():
    num_points = 3
    max_derivative = 3
    # Combinations of (family, dim, degree) to test
    test_cases = (
        ("Lagrange", 2, 1),
        ("Lagrange", 2, 2),
        ("Lagrange", 2, 3),
        ("Lagrange", 3, 1),
        ("Lagrange", 3, 2),
        ("Lagrange", 3, 3),
        ("Discontinuous Lagrange", 2, 0),
        ("Discontinuous Lagrange", 2, 1),
        ("Discontinuous Lagrange", 2, 2),
        ("Discontinuous Lagrange", 3, 0),
        ("Discontinuous Lagrange", 3, 1),
        ("Discontinuous Lagrange", 3, 2),
        ("Brezzi-Douglas-Marini", 2, 1),
        ("Brezzi-Douglas-Marini", 2, 2),
        ("Brezzi-Douglas-Marini", 2, 3),
        ("Brezzi-Douglas-Marini", 3, 1),
        ("Brezzi-Douglas-Marini", 3, 2),
        ("Brezzi-Douglas-Marini", 3, 3),
        ("Brezzi-Douglas-Fortin-Marini", 2, 2),
        ("Raviart-Thomas", 2, 1),
        ("Raviart-Thomas", 2, 2),
        ("Raviart-Thomas", 2, 3),
        ("Raviart-Thomas", 3, 1),
        ("Raviart-Thomas", 3, 2),
        ("Raviart-Thomas", 3, 3),
        ("Discontinuous Raviart-Thomas", 2, 1),
        ("Discontinuous Raviart-Thomas", 2, 2),
        ("Discontinuous Raviart-Thomas", 2, 3),
        ("Discontinuous Raviart-Thomas", 3, 1),
        ("Discontinuous Raviart-Thomas", 3, 2),
        ("Discontinuous Raviart-Thomas", 3, 3),
        ("Nedelec 1st kind H(curl)", 2, 1),
        ("Nedelec 1st kind H(curl)", 2, 2),
        ("Nedelec 1st kind H(curl)", 2, 3),
        ("Nedelec 1st kind H(curl)", 3, 1),
        ("Nedelec 1st kind H(curl)", 3, 2),
        ("Nedelec 1st kind H(curl)", 3, 3),
        ("Nedelec 2nd kind H(curl)", 2, 1),
        ("Nedelec 2nd kind H(curl)", 2, 2),
        ("Nedelec 2nd kind H(curl)", 2, 3),
        ("Nedelec 2nd kind H(curl)", 3, 1),
        ("Nedelec 2nd kind H(curl)", 3, 2),
        ("Nedelec 2nd kind H(curl)", 3, 3),
        ("Crouzeix-Raviart", 2, 1),
        ("Crouzeix-Raviart", 3, 1)
        )

    def create_data(family, dim, degree):
        '''Create the reference data.
        '''
        # Get domain and element class
        domain = ufc_simplex(dim)
        ElementClass = supported_elements[family]
        # Create element|
        element = ElementClass(domain, degree)
        # Create quadrature points
        quad_rule = make_quadrature(domain, num_points)
        points = quad_rule.get_points()
        # Tabulate at quadrature points
        table = element.tabulate(max_derivative, points)
        return table

    def _perform_test(family, dim, degree, reference_table):
        '''Test against reference data.
        '''
        table = create_data(family, dim, degree)
        # Check against reference
        for dtuple in reference_table:
            assert eval(dtuple) in table
            assert table[eval(dtuple)].shape == reference_table[dtuple].shape
            diff = table[eval(dtuple)] - reference_table[dtuple]
            assert (abs(diff) < tolerance).all()
        return

    # Try reading reference values
    filename = "reference.json"

    try:
        reference = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
    except IOError:
        reference = {}
        for test_case in test_cases:
            family, dim, degree = test_case
            ref = dict([(str(k), v) for k, v in create_data(family, dim, degree).items()])
            reference[str(test_case)] = ref
        # Store the data for the future
        json.dump(reference, open(filename, "w"), cls=NumpyEncoder)

    for test_case in test_cases:
        family, dim, degree = test_case
        yield _perform_test, family, dim, degree, reference[str(test_case)]


if __name__ == "__main__":
    nose.main()
