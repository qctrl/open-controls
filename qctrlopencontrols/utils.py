# Copyright 2022 Q-CTRL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Miscellaneous functions and dataclasses.
"""
from __future__ import annotations

from enum import Enum
from typing import (
    Any,
    Optional,
)

from .exceptions import ArgumentsValueError


def create_repr_from_attributes(class_instance=None, attributes=None) -> str:

    """
    Returns a string representation of an object.

    Parameters
    ----------
    class_instance : object, optional
        The instance of a class (object)l defaults to None
    attributes : list, optional
        A list of string where each entry is the name of the attribute to collect
        from the class instance.

    Returns
    -------
    str
        A string representing the attributes; If no attribute is provided
        a constant string is returned
        'No attributes provided for object of class {0.__class__.__name__}".
        format(class_instance)'

    Raises
    ------
    ArgumentsValueError
        If class name is not a string or any of the attribute name is not string type
    """

    if class_instance is None:
        raise ArgumentsValueError(
            "Class instance must be a valid object.", {"class_instance": class_instance}
        )

    class_name = f"{class_instance.__class__.__name__}("

    if attributes is None:
        raise ArgumentsValueError(
            "Attributes must be a list of string", {"attributes": attributes}
        )

    if not attributes:
        return f"No attributes provided for object of class {class_name}"

    for attribute in attributes:
        if not isinstance(attribute, str):
            raise ArgumentsValueError(
                "Each attribute name must be a string. Found "
                f"{type(attribute)} type.",
                {"attribute": attribute, "type(attribute)": type(attribute)},
            )

    repr_string = f"{class_name}"
    attributes_string = ",".join(
        f"{attribute}={repr(getattr(class_instance, attribute))}"
        for attribute in attributes
    )
    repr_string += attributes_string
    repr_string += ")"

    return repr_string


def check_arguments(
    condition: Any,
    description: str,
    arguments: dict[str, Any],
    extras: Optional[dict[str, Any]] = None,
):
    """
    Raises an ArgumentsValueError with the specified parameters if the given condition is false,
    otherwise does nothing.

    For example, a use case may look like::

        def log(x):
            check_arguments(
                x > 0, "x must be positive.", {"x": x}
            )
            return np.log(x)

    Parameters
    ----------
    condition : Any
        The condition to be checked. Evaluated result of the condition must be bool.
    description : str
        Error information to explain why condition fails.
    arguments : dict
        arguments that fail the condition. Keys should be the names of the arguments and arguments
        are the values.
    extras : dict, optional
        Any extra information to explain why condition fails. Defaults to None.

    Raises
    ------
    ArgumentsValueError
        If condition is false.
    """
    if condition:
        return
    raise ArgumentsValueError(description, arguments, extras=extras)


class FileFormat(Enum):
    """
    Defines exported file format.

    Currently only supports the Q-CTRL expanded format. See :py:meth:`DrivenControl.export_to_file`
    for details.
    """

    QCTRL = "Q-CTRL expanded"


class FileType(Enum):
    """
    Defines exported file type.
    """

    JSON = "JSON"
    CSV = "CSV"


class Coordinate(Enum):
    """
    Defines coordinate system for data representation.
    """

    CARTESIAN = "cartesian"
    CYLINDRICAL = "cylindrical"
