# Copyright 2019 Q-CTRL Pty Ltd & Q-CTRL Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module with helper methods to check for third party packages.
"""

import pkg_resources
from pkg_resources import DistributionNotFound
from packaging.version import parse


from qctrlopencontrols.exceptions.exceptions import (
    PackageVersionMismatchError,
    PackageNotFoundError)


def check_package(package_name, minimum_package_version):
    """Checks if the specified package and its minimum version
    exists.

    Parameters
    ---------
    package_name: str
        The name of the package
    minimum_package_version: str
        The minimum version of the package
    """

    try:
        found_version = pkg_resources.get_distribution(package_name).version
        if parse(found_version) < parse(minimum_package_version):
            raise PackageVersionMismatchError(
                'Minimum version of the {} package not found.'.format(package_name),
                {'package name': package_name,
                 'minimum package version': minimum_package_version},
                extras={'version found': found_version})
    except DistributionNotFound:
        raise PackageNotFoundError(
            'Package {} not found.'.format(package_name),
            {'package name': package_name,
             'minimum package version': minimum_package_version})
