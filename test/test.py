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
# Last changed: 2014-06-23

import nose

import pickle
import numpy

from FIAT import supported_elements, make_quadrature, ufc_simplex

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

# Parameters
num_points = 3
max_derivative = 3
tolerance = 1e-8


def test_generator():
    # Try reading reference values
    try:
        reference = pickle.load(open("reference.pickle", "r"))
    except IOError:
        print("Creating new reference values")
        reference = {}
        for test_case in test_cases:
            family, dim, degree = test_case
            reference[test_case] = _create_data(family, dim, degree)
        # Store the data for the future
        pickle.dump(reference, open("reference.pickle", "w"))

    for test_case in test_cases:
        family, dim, degree = test_case
        yield _perform_test, family, dim, degree, reference[test_case]


def _create_data(family, dim, degree):
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
    table = _create_data(family, dim, degree)

    # Check against reference
    for dtuple in reference_table:
        assert dtuple in table
        assert table[dtuple].shape == reference_table[dtuple].shape
        diff = numpy.amax(abs(table[dtuple] - reference_table[dtuple]))
        assert diff < tolerance

    return


if __name__ == "__main__":
    nose.main()
