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
from qctrlopencontrols import DrivenControl
from qctrlopencontrols.globals import CARTESIAN, CYLINDRICAL


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

    driven_control = DrivenControl(
        segments=_segments, name=_name)

    assert np.allclose(driven_control.segments, _segments)
    assert driven_control.number_of_segments == 3
    assert np.allclose(driven_control.segment_durations, np.array(
        [1., 2., 3.]))

    assert driven_control.name == _name

    with pytest.raises(ArgumentsValueError):

        _ = DrivenControl(segments=[[1e12, 0., 3, 1.]])
        _ = DrivenControl(segments=[[3., 0., 1e12, 1.]])
        _ = DrivenControl(segments=[[3., 0., 1e12, -1.]])
        _ = DrivenControl(segments=[[0., 0., 0., 0.]])


def test_control_export():

    """Tests exporting the control to a file
    """

    _maximum_rabi_rate = 5*np.pi
    _segments = [[_maximum_rabi_rate*np.cos(np.pi/4), _maximum_rabi_rate*np.sin(np.pi/4), 0., 2.],
                 [_maximum_rabi_rate*np.cos(np.pi/3), _maximum_rabi_rate*np.sin(np.pi/3), 0., 2.],
                 [0., 0., np.pi, 1.]]
    _name = 'driven_controls'

    driven_control = DrivenControl(
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

def test_plot_data():
    """
    Test the plot data produced for a driven control.
    """

    segments = [[1., 0., 0., 2.],
                [0., 1.5, 1.7, 3.],
                [1., 0., 2.1, 0.5]]
    x_amplitude = [0., 1., 1., 0., 0., 1., 1., 0.]
    y_amplitude = [0., 0., 0., 1.5, 1.5, 0., 0., 0.]
    z_amplitude = [0., 0., 0., 1.7, 1.7, 2.1, 2.1, 0.]
    times = [0., 0., 2., 2., 5., 5., 5.5, 5.5]
    driven_control = DrivenControl(segments=segments)
    plot_data = driven_control.get_plot_formatted_arrays(dimensionless_rabi_rate=False)

    assert np.allclose(plot_data['times'], times)
    assert np.allclose(plot_data['amplitude_x'], x_amplitude)
    assert np.allclose(plot_data['amplitude_y'], y_amplitude)
    assert np.allclose(plot_data['detuning'], z_amplitude)

def test_dimensionless_segments():
    """
    Test the dimensionless amplitude and angle segments generated
    """
    segments = [[1., 0., 0., np.pi / 2],
                [0., 1., 0., np.pi / 2],
                [1. / np.sqrt(2.), 0., 1. / np.sqrt(2.), np.pi / 2]]

    _on_resonance_amplitudes = np.array([1., 1., 1. / np.sqrt(2.)])
    _azimuthal_angles = np.array([0., np.pi / 2, 0.])
    _detunings = np.array([0, 0, 1. / np.sqrt(2.)])
    _durations = np.pi / 2. * np.array([1., 1., 1.])

    amplitude_angle_segments = np.stack((_on_resonance_amplitudes, _azimuthal_angles,
                                         _detunings, _durations), axis=1)

    driven_control = DrivenControl(segments=segments)
    _max_rabi = driven_control.maximum_rabi_rate

    dimensionless_euclid = segments.copy()
    dimensionless_euclid = np.array(dimensionless_euclid)
    dimensionless_euclid[:, 0:2] = dimensionless_euclid[:, 0:2] / _max_rabi

    dimensionless_cylinder = amplitude_angle_segments.copy()
    dimensionless_cylinder = np.array(dimensionless_cylinder)
    dimensionless_cylinder[:, 0] = dimensionless_cylinder[:, 0] / _max_rabi

    transformed_euclidean = driven_control.get_transformed_segments(coordinates=CARTESIAN,
                                                                    dimensionless_rabi_rate=False)

    assert np.allclose(segments, transformed_euclidean)

    transformed_euclidean = driven_control.get_transformed_segments(coordinates=CARTESIAN,
                                                                    dimensionless_rabi_rate=True)

    assert np.allclose(dimensionless_euclid, transformed_euclidean)

    transformed_cylindrical = driven_control.get_transformed_segments(coordinates=CYLINDRICAL,
                                                                      dimensionless_rabi_rate=False)

    assert np.allclose(amplitude_angle_segments, transformed_cylindrical)

    transformed_cylindrical = driven_control.get_transformed_segments(coordinates=CYLINDRICAL,
                                                                      dimensionless_rabi_rate=True)

    assert np.allclose(amplitude_angle_segments, transformed_cylindrical)
