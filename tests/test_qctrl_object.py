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
Tests for base module
=====================
"""

import pytest
from qctrlopencontrols.exceptions import ArgumentsValueError
from qctrlopencontrols.base import QctrlObject


class SampleClass(QctrlObject):  #pylint: disable=too-few-public-methods

    """A sample class with attributes

    Parameters
    ----------
    sample_attribute : int
        A sample attribute of integer type
    base_attributes : list
        A list of attributes to be used as base_attributes
    """

    def __init__(self, sample_attribute, base_attributes):
        super(SampleClass, self).__init__(base_attributes=base_attributes)
        self.sample_attribute = sample_attribute


def test_qctrl_object():    # pylint: disable=too-few-public-methods
    """Tests the __repr__ and __str__ methods of base.QctrlObject
    """

    sample_class = SampleClass(sample_attribute=50, base_attributes=['sample_attribute'])

    _sample_repr = '{0.__class__.__name__!s}(sample_attribute={0.sample_attribute!r})'.format(
        sample_class)

    assert repr(sample_class) == _sample_repr
    assert str(sample_class) == str(_sample_repr)

    sample_class = SampleClass(sample_attribute=50, base_attributes=None)
    _sample_repr = 'No attributes provided for object of class {0.__class__.__name__!s}'.format(
        sample_class)

    assert repr(sample_class) == _sample_repr
    assert str(sample_class) == str(_sample_repr)

    with pytest.raises(ArgumentsValueError):

        _ = SampleClass(sample_attribute=50., base_attributes=[])
        _ = SampleClass(sample_attribute=50., base_attributes=['sample_1', 40])
        _ = SampleClass(sample_attribute=50., base_attributes='no list')
