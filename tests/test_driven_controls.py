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

from qctrlopencontrols.driven_controls.constants import (
    UPPER_BOUND_SEGMENTS, UPPER_BOUND_RABI_RATE, UPPER_BOUND_DETUNING_RATE)


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
    _rabi_rates = [np.pi, np.pi, 0]
    _azimuthal_angles = [np.pi/2, 0, -np.pi]
    _detunings = [0, 0, 0]
    _durations = [1, 2, 3]

    _name = 'driven_control'

    driven_control = DrivenControl(
        rabi_rates=_rabi_rates,
        azimuthal_angles=_azimuthal_angles,
        detunings=_detunings,
        durations=_durations,
        name=_name)

    assert np.allclose(driven_control.rabi_rates, _rabi_rates)
    assert np.allclose(driven_control.durations, _durations)
    assert np.allclose(driven_control.detunings, _detunings)
    assert np.allclose(driven_control.azimuthal_angles, _azimuthal_angles)

    assert driven_control.name == _name

    with pytest.raises(ArgumentsValueError):
        _ = DrivenControl(rabi_rates=[-1])
    with pytest.raises(ArgumentsValueError):
        _ = DrivenControl(durations=[0])
    with pytest.raises(ArgumentsValueError):
        _ = DrivenControl(rabi_rates=[1.1 * UPPER_BOUND_RABI_RATE])
    with pytest.raises(ArgumentsValueError):
        _ = DrivenControl(detunings=[1.1 * UPPER_BOUND_DETUNING_RATE])
    with pytest.raises(ArgumentsValueError):
        _ = DrivenControl(rabi_rates=[1] * UPPER_BOUND_SEGMENTS + [1])

def test_control_export():

    """Tests exporting the control to a file
    """
    _rabi_rates = [5 * np.pi, 4 * np.pi, 3 * np.pi]
    _azimuthal_angles = [np.pi / 4, np.pi / 3, 0]
    _detunings = [0, 0, np.pi]
    _durations = [2, 2, 1]

    _name = 'driven_control'

    driven_control = DrivenControl(
        rabi_rates=_rabi_rates,
        azimuthal_angles=_azimuthal_angles,
        detunings=_detunings,
        durations=_durations,
        name=_name
    )

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
    _rabi_rates = [np.pi, 2 * np.pi, np.pi]
    _azimuthal_angles = [0, np.pi/2, -np.pi/2]
    _detunings = [0, 1, 0]
    _durations = [1, 1.25, 1.5]

    driven_control = DrivenControl(
        rabi_rates=_rabi_rates,
        azimuthal_angles=_azimuthal_angles,
        detunings=_detunings,
        durations=_durations
    )

    x_amplitude = [0., np.pi, np.pi, 0., 0., 0., 0., 0.]
    y_amplitude = [0., 0., 0., 2*np.pi, 2*np.pi, -np.pi, -np.pi, 0.]
    z_amplitude = [0., 0., 0., 1., 1., 0., 0., 0.]
    times = [0., 0., 1., 1., 2.25, 2.25, 3.75, 3.75]

    plot_data = driven_control.get_plot_formatted_arrays(
        dimensionless_rabi_rate=False, coordinates='cartesian'
    )

    assert np.allclose(plot_data['times'], times)
    assert np.allclose(plot_data['amplitudes_x'], x_amplitude)
    assert np.allclose(plot_data['amplitudes_y'], y_amplitude)
    assert np.allclose(plot_data['detunings'], z_amplitude)
