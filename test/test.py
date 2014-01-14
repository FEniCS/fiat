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
# Last changed: 2012-10-01

import pickle
from FIAT import supported_elements, make_quadrature, ufc_simplex
from numpy import shape, max, abs

# Combinations of (family, dim, degree) to test
test_cases = (("Lagrange", 2, 1),
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
              ("Crouzeix-Raviart", 3, 1))

# Parameters
num_points = 3
max_derivative = 3
tolerance = 1e-8

def test():
    "Regression test all elements."

    # Try reading reference values
    try:
        reference = pickle.load(open("reference.pickle", "r"))
    except:
        reference = None

    # Iterate over test cases
    values = {}
    for test_case in test_cases:

        print("Testing", test_case)

        # Get family, dimension and degree
        family, dim, degree = test_case

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
        values[test_case] = table

        # Check against reference
        if reference is not None:
            reference_table = reference[(family, dim, degree)]
            for dtuple in reference_table:
                if dtuple not in table:
                    print("*** Missing dtuple in table for " + str(test_case))
                    return 1
                elif shape(table[dtuple]) != shape(reference_table[dtuple]):
                    print("*** Wrong shape in table for " + str(test_case))
                    return 1
                else:
                    diff = max(abs(table[dtuple] - reference_table[dtuple]))
                    if diff > tolerance:
                        print("*** Wrong values in table for %s, difference is %g" % (str(test_case), diff))
                        return 1

    # Write new values if reference is missing
    if reference is None:
        print("Storing new reference values")
        pickle.dump(values, open("reference.pickle", "w"))

    print()
    print("Ran %d tests: OK" % len(test_cases))

    return 0

if __name__ == "__main__":
    import sys, TFEtest
    sys.exit(test() or TFEtest.test())
