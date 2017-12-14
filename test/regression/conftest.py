# Copyright (C) 2016 Jan Blechta
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

import os


# Directories
path = os.path.dirname(os.path.abspath(__file__))
ref_path = os.path.join(path, 'fiat-reference-data')
download_script = os.path.join(path, 'scripts', 'download')


def pytest_configure(config):
    # Download reference data
    if config.getoption("download"):
        failure = download_reference()
        if failure:
            raise RuntimeError("Download reference data failed")
        print("Download reference data ok")
    else:
        print("Skipping reference data download")
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)


def download_reference():
    _path = os.getcwd()
    os.chdir(path)
    rc = os.system(download_script)
    os.chdir(_path)
    return rc
