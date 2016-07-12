"""Run all tests, including unit tests and regression tests"""

# Copyright (C) 2007 Anders Logg
#
# This file is part of fiat.
#
# fiat is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# fiat is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with fiat. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-06-09
# Last changed: 2016-07-08

import os
import sys

# Name of log file
pwd = os.path.dirname(os.path.abspath(__file__))
logfile = os.path.join(pwd, "test.log")
os.system("rm -f %s" % logfile)

# Tests to run
tests = [("unit", "py.test"), ("regression", "python test.py")]

# Run tests
failed = []
for test, command in tests:
    print("Running tests: %s" % test)
    print("----------------------------------------------------------------------")
    os.chdir(os.path.join(pwd, test))
    # failure = os.system("python test.py | tee -a %s" % logfile)
    failure = os.system(command)
    if failure:
        print("Test FAILED")
        failed.append(test)
    print("")

# print("To view the test log, use the following command: less -R test.log")

sys.exit(len(failed))
