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
==========
base.utils
==========
"""

from qctrlopencontrols.exceptions import ArgumentsValueError


def create_repr_from_attributes(class_name, **attributes):

    """Returns a string representation of an object

    Parameters
    ----------
    class_name : str
        The name of the class
    attributes : dict
        A dict of attributes in (attribute_name:attribute_value) format

    Returns
    -------
    str
        A string representing the attributes

    Raises
    ------
    ArgumentsValueError
        If no class name is provided or any of the attribute name is not string type
    """

    if not attributes:
        return "No attributes provided for object of class {0}".format(class_name)

    for attribute in attributes:
        if not isinstance(attribute, str):
            raise ArgumentsValueError('Each attribute name must be a string. Found '
                                      '{0} type.'.format(type(attribute)),
                                      {'attribute': attribute,
                                       'type(attribute)': type(attribute)})

    repr_string = '{0}('.format(class_name)
    attributes_string = ','.join('{0}={1}'.format(attribute,
                                                  repr(attributes[attribute]))
                                 for attribute in attributes)
    repr_string += attributes_string
    repr_string += ')'

    return repr_string
