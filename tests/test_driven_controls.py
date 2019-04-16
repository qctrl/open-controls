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
Test for driven controls
"""

import os

import numpy as np
import pytest

from qctrlopencontrols.exceptions import ArgumentsValueError
from qctrlopencontrols import DrivenControls


def _remove_file(filename):
    """Removes the file after test done
    """

    if os.path.exists(filename):
        os.remove(filename)
    else:
        raise IOError('Could not find file {}'.format(
            filename))


def test_driven_controls():

    """Tests the construction of driven controls
    """
    _segments = [[np.pi, 0., 0., 1.],
                 [np.pi, np.pi/2, 0., 2.],
                 [0., 0., np.pi, 3.]]

    _name = 'driven_control'

    driven_control = DrivenControls(
        segments=_segments, name=_name)

    assert np.allclose(driven_control.segments, _segments)
    assert driven_control.number_of_segments == 3
    assert np.allclose(driven_control.segment_durations, np.array(
        [1., 2., 3.]))

    assert driven_control.name == _name

    with pytest.raises(ArgumentsValueError):

        _ = DrivenControls(segments=[[1e12, 0., 3, 1.]])
        _ = DrivenControls(segments=[[3., 0., 1e12, 1.]])
        _ = DrivenControls(segments=[[3., 0., 1e12, -1.]])
        _ = DrivenControls(segments=[[0., 0., 0., 0.]])


def test_control_export():

    """Tests exporting the control to a file
    """

    _maximum_rabi_rate = 5*np.pi
    _segments = [[_maximum_rabi_rate*np.cos(np.pi/4), _maximum_rabi_rate*np.sin(np.pi/4), 0., 2.],
                 [_maximum_rabi_rate*np.cos(np.pi/3), _maximum_rabi_rate*np.sin(np.pi/3), 0., 2.],
                 [0., 0., np.pi, 1.]]
    _name = 'driven_controls'

    driven_control = DrivenControls(
        segments=_segments, name=_name)

    _filename = 'driven_control_qctrl_cylindrical.csv'
    driven_control.export_to_file(
        filename=_filename,
        file_format='Q-CTRL expanded',
        file_type='CSV',
        coordinates='cylindrical')

    _filename = 'driven_control_qctrl_cartesian.csv'
    driven_control.export_to_file(
        filename=_filename,
        file_format='Q-CTRL expanded',
        file_type='CSV',
        coordinates='cartesian')

    _filename = 'driven_control_qctrl_cylindrical.json'
    driven_control.export_to_file(
        filename=_filename,
        file_format='Q-CTRL expanded',
        file_type='JSON',
        coordinates='cylindrical')

    _filename = 'driven_control_qctrl_cartesian.json'
    driven_control.export_to_file(
        filename=_filename,
        file_format='Q-CTRL expanded',
        file_type='JSON',
        coordinates='cartesian')

    _remove_file('driven_control_qctrl_cylindrical.csv')
    _remove_file('driven_control_qctrl_cartesian.csv')
    _remove_file('driven_control_qctrl_cylindrical.json')
    _remove_file('driven_control_qctrl_cartesian.json')
