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
=================
base.qctrl_object
=================
"""

from qctrlopencontrols.exceptions import ArgumentsValueError


class QctrlObject(object):
    """Base class for all classes in QCtrl library.

    Parameters
    ----------
    base_attributes : list, optional
        List of names of attributes. Defaults to None

    Raises
    ------
    ArgumentsValueError
        If base_attributes is not list type or empty list
        If any of the base_attributes is not str type

    Notes
    -----
    If the base_attributes is None, __repr__ and __str__
    return a default string "No attributes provided for object
    of class self.__class__.__name__"
    """

    def __init__(self, base_attributes=None):

        self.base_attributes = base_attributes

        if self.base_attributes is not None:

            if not isinstance(base_attributes, list):

                raise ArgumentsValueError('Attributes must be provided as a list object',
                                          {'base_attributes': self.base_attributes},
                                          extras={'base_attributes_type': type(base_attributes)})

            if not self.base_attributes:
                raise ArgumentsValueError('No attributes provided',
                                          {'base_attributes': self.base_attributes})

            for attribute in self.base_attributes:

                if not isinstance(attribute, str):
                    raise ArgumentsValueError('Each attribute must be a string. Found '
                                              '{0} type.'.format(type(attribute)),
                                              {'attribute': attribute,
                                               'attribute_type': type(attribute)},
                                              extras={'base_attributes': self.base_attributes})

    def __repr__(self):
        """The returned string looks like a valid Python expression that could be used
        to recreate the object, including default arguments.


        Returns
        -------
        str
            String representation of the object including the values of the arguments.
            However, if the base_attributes is None, return a fixed string "No attributes
            provided for object of class self.__class__.__name__"
        """

        if self.base_attributes is None:

            return "No attributes provided for object of class {0.__class__.__name__!s}".format(
                self)

        repr_string = '{0.__class__.__name__!s}('.format(self)

        attributes_string = ','.join('{0}={1}'.format(attribute,
                                                      repr(getattr(self, attribute)))
                                     for attribute in self.base_attributes)
        repr_string += attributes_string
        repr_string += ')'

        return repr_string

    def __str__(self):
        """Returns a string representation of the object.

        Returns
        -------
        str
            The object definition as a string
        """

        return str(self.__repr__())
