# Copyright (C) 2010 Anders Logg, 2015 Jan Blechta
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

import pytest
import json
import numpy
import warnings
import os

from FIAT import supported_elements, make_quadrature, ufc_simplex, \
    expansions, reference_element, polynomial_set

# Parameters
tolerance = 1e-8

# Directories
path = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(path, 'fiat-reference-data')
download_script = os.path.join(path, 'scripts', 'download')


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


def load_reference(filename, create_data):
    """Load reference from file. On failure create new file using supplied
    function.
    """
    try:
        # Try loading the reference
        reference = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
    except IOError:
        warnings.warn('Reference file "%s" could not be loaded! '
                      'Creating a new reference file!' % filename,
                      RuntimeWarning)

        # Generate data and store for the future
        reference = create_data()
        json.dump(reference, open(filename, "w"), cls=NumpyEncoder)

        # Report failure
        pytest.fail('Comparison to "%s" failed!' % filename)

    return reference


def test_polynomials():
    def create_data():
        ps = polynomial_set.ONPolynomialSet(
            ref_el=reference_element.DefaultTetrahedron(),
            degree=3
        )
        return ps.dmats

    # Try reading reference values
    filename = os.path.join(ref_path, "reference-polynomials.json")
    reference = load_reference(filename, create_data)

    dmats = create_data()

    for dmat, reference_dmat in zip(dmats, reference):
        assert (abs(dmat - reference_dmat) < tolerance).all()


def test_polynomials_1D():
    def create_data():
        ps = polynomial_set.ONPolynomialSet(
            ref_el=reference_element.DefaultLine(),
            degree=3
        )
        return ps.dmats

    # Try reading reference values
    filename = os.path.join(ref_path, "reference-polynomials_1D.json")
    reference = load_reference(filename, create_data)

    dmats = create_data()

    for dmat, reference_dmat in zip(dmats, reference):
        assert (abs(dmat - reference_dmat) < tolerance).all()


def test_expansions():
    def create_data():
        E = reference_element.DefaultTriangle()
        k = 3
        pts = reference_element.make_lattice(E.get_vertices(), k)
        Phis = expansions.get_expansion_set(E)
        phis = Phis.tabulate(k, pts)
        dphis = Phis.tabulate_derivatives(k, pts)
        return phis, dphis

    # Try reading reference values
    filename = os.path.join(ref_path, "reference-expansions.json")
    reference = load_reference(filename, create_data)

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


def test_expansions_jet():
    def create_data():
        latticeK = 2
        n = 1
        order = 2
        E = reference_element.DefaultTetrahedron()
        pts = reference_element.make_lattice(E.get_vertices(), latticeK)
        F = expansions.TetrahedronExpansionSet(E)
        return F.tabulate_jet(n, pts, order)

    filename = os.path.join(ref_path, "reference-expansions-jet.json")
    reference = load_reference(filename, create_data)

    # Test jet data
    data = create_data()
    for datum, reference_datum in zip(data, reference):
        diff = numpy.array(datum) - numpy.array(reference_datum)
        assert (abs(diff) < tolerance).all()


def test_quadrature():
    num_points = 3
    max_derivative = 3
    # Combinations of (family, dim, degree) to test
    test_cases = (
        ("Lagrange", 1, 1),
        ("Lagrange", 1, 2),
        ("Lagrange", 1, 3),
        ("Lagrange", 2, 1),
        ("Lagrange", 2, 2),
        ("Lagrange", 2, 3),
        ("Lagrange", 3, 1),
        ("Lagrange", 3, 2),
        ("Lagrange", 3, 3),
        ("Discontinuous Lagrange", 1, 0),
        ("Discontinuous Lagrange", 1, 1),
        ("Discontinuous Lagrange", 1, 2),
        ("Discontinuous Lagrange", 2, 0),
        ("Discontinuous Lagrange", 2, 1),
        ("Discontinuous Lagrange", 2, 2),
        ("Discontinuous Lagrange", 3, 0),
        ("Discontinuous Lagrange", 3, 1),
        ("Discontinuous Lagrange", 3, 2),
        ("Discontinuous Taylor", 1, 0),
        ("Discontinuous Taylor", 1, 1),
        ("Discontinuous Taylor", 1, 2),
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
        ("Crouzeix-Raviart", 1, 1),
        ("Crouzeix-Raviart", 2, 1),
        ("Crouzeix-Raviart", 3, 1),
        ("Regge", 2, 0),
        ("Regge", 2, 1),
        ("Regge", 2, 2),
        ("Regge", 3, 0),
        ("Regge", 3, 1),
        ("Regge", 3, 2),
        ("Bubble", 2, 3),
        ("Bubble", 2, 4),
        ("Bubble", 2, 5),
        ("Bubble", 3, 4),
        ("Bubble", 3, 5),
        ("Bubble", 3, 6),
        ("Hellan-Herrmann-Johnson", 2, 0),
        ("Hellan-Herrmann-Johnson", 2, 1),
        ("Hellan-Herrmann-Johnson", 2, 2),
    )

    def create_data(family, dim, degree):
        '''Create the reference data.
        '''
        # Get domain and element class
        domain = ufc_simplex(dim)
        ElementClass = supported_elements[family]
        # Create element
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
            assert (abs(diff) < tolerance).all(), \
                "quadrature case %s %s %s failed!" % (family, dim, degree)

    filename = os.path.join(ref_path, "reference.json")

    # Try comparing against references
    try:
        reference = json.load(open(filename, "r"), object_hook=json_numpy_obj_hook)
        for test_case in test_cases:
            family, dim, degree = test_case
            yield _perform_test, family, dim, degree, reference[str(test_case)]

    # Update references if missing
    except (IOError, KeyError) as e:
        if isinstance(e, IOError):
            warnings.warn('Reference file "%s" could not be loaded! '
                          'Creating a new reference file!' % filename,
                          RuntimeWarning)
        else:
            assert isinstance(e, KeyError) and len(e.args) == 1
            warnings.warn('Reference file "%s" does not contain reference "%s"! '
                          'Creating a new reference file!'
                          % (filename, e.args[0]), RuntimeWarning)
        reference = {}
        for test_case in test_cases:
            family, dim, degree = test_case
            ref = dict([(str(k), v) for k, v in create_data(family, dim, degree).items()])
            reference[str(test_case)] = ref
        # Store the data for the future
        json.dump(reference, open(filename, "w"), cls=NumpyEncoder)

        # Report failure
        pytest.fail('Comparison to "%s" failed!' % filename)


if __name__ == '__main__':
    pytest.main(os.path.abspath(__file__))
