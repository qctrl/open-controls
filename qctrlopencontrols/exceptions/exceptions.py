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
=====================
exceptions.exceptions
=====================
"""


class QctrlError(Exception):
    """
    Base class for exceptions raised by QCtrl package.
    """
    pass  # pylint: disable=unnecessary-pass


class ArgumentsValueError(QctrlError):
    """
    Exception thrown when one or more arguments provided to a method have incorrect values.

    Notes
    -----
    If the error message passed to the console is a combination of the problem, then
    the name of each attribute and its corresponding representation (repr(attribute)). Once
    constructed the attribute message contains the full error message.

    Parameters
    ----------
    description : string
        Description of the why the input error occurred.
    arguments : dict
        Dictionary that can only contain the arguments of the method that contributed to the error.
    extras : dict
        Other variables that contributed to the error but are not attributes of the method.
    """

    def __init__(self, description, arguments, extras=None):
        self.description = description
        self.arguments = arguments
        self.extras = extras
        self.message = self.description
        for key in self.arguments:
            self.message += '\n' + str(key) + '=' + repr(self.arguments[key])
        if extras is not None:
            for key in self.extras:
                self.message += '\n' + str(key) + '=' + repr(self.extras[key])
        super(ArgumentsValueError, self).__init__(self.message)


class PackageVersionMismatchError(QctrlError):
    """Exception raised when the version of required third party packages found but is
    below the minimum required.

    Parameters
    ----------
    description : string
        Description of the why the error is raised.
    arguments : dict
        Dictionary that contains the package name, required minimum version.
    extras : dict, optional
        Other variables, such as the version found in the system,
        that may be relevant to the error. Defaults to None.
    """

    def __init__(self, description, arguments, extras=None):
        self.description = description
        self.arguments = arguments
        self.extras = extras
        self.message = self.description

        for key in self.arguments:
            self.message += '\n' + str(key) + '=' + repr(self.arguments[key])
        if extras is not None:
            for key in self.extras:
                self.message += '\n' + str(key) + '=' + repr(self.extras[key])
        super(PackageVersionMismatchError, self).__init__(self.message)


class PackageNotFoundError(QctrlError):
    """Exception raised when the required third party packages is not found.

    Parameters
    ----------
    description : string
        Description of the why the error is raised.
    arguments : dict
        Dictionary that contains the package name and the required minimum version.
    extras : dict, optional
        Other variables that may be relevant to the error. Defaults to None.
    """

    def __init__(self, description, arguments, extras=None):
        self.description = description
        self.arguments = arguments
        self.extras = extras
        self.message = self.description

        for key in self.arguments:
            self.message += '\n' + str(key) + '=' + repr(self.arguments[key])
        if extras is not None:
            for key in self.extras:
                self.message += '\n' + str(key) + '=' + repr(self.extras[key])
        super(PackageNotFoundError, self).__init__(self.message)


class PackageImportError(QctrlError):
    """Exception raised when a package cannot be imported.

    Parameters
    ----------
    description : string
        Description of the why the error is raised.
    package_name : string, optional
        The name of the package that could not be imported. Defaults to None.
    """
    def __init__(self, description, package_name=None):
        self.description = description
        self.package_name = package_name
        self.message = self.description

        if package_name:
            self.message += '\n' + 'package' + '=' + repr(self.arguments[package_name])
        super(PackageImportError, self).__init__(self.message)
