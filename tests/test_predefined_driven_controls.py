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
====================================
Tests for Predefined Driven Controls
====================================
"""
import numpy as np
import pytest

from qctrlopencontrols.exceptions import ArgumentsValueError

from qctrlopencontrols.driven_controls import (
    new_predefined_driven_control,
    new_primitive_control, new_wimperis_1_control, new_solovay_kitaev_1_control,
    new_compensating_for_off_resonance_with_a_pulse_sequence_control,
    new_compensating_for_off_resonance_with_a_pulse_sequence_with_solovay_kitaev_control,
    new_compensating_for_off_resonance_with_a_pulse_sequence_with_wimperis_control,
    new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control,
    new_walsh_amplitude_modulated_filter_1_control,
    new_corpse_in_scrofulous_control
)

from qctrlopencontrols.globals import SQUARE


def test_new_predefined_driven_control():
    """Test the new_predefined_driven_control function in
       qctrlopencontrols.driven_controls.predefined
    """
    # Test that an error is raised if supplied with an unknown scheme
    with pytest.raises(ArgumentsValueError):
        _ = new_predefined_driven_control(driven_control_type='nil')


def test_primitive_control_segments():
    """Test the segments of the predefined primitive driven control
    """
    _rabi_rate = 1
    _rabi_rotation = np.pi
    _azimuthal_angle = np.pi/2
    _segments = [[
        np.cos(_azimuthal_angle),
        np.sin(_azimuthal_angle),
        0.,
        _rabi_rotation], ]

    primitive_control = new_primitive_control(
        rabi_rotation=_rabi_rotation,
        maximum_rabi_rate=_rabi_rate,
        azimuthal_angle=_azimuthal_angle,
        shape=SQUARE
    )

    assert np.allclose(_segments, primitive_control.segments)

def test_new_wimperis_1_control():
    """Test the segments of the Wimperis 1 (BB1) driven control
    """
    _rabi_rotation = np.pi
    _azimuthal_angle = np.pi/2

    phi_p = np.arccos(-_rabi_rotation / (4 * np.pi))

    _segments = [
        [np.cos(_azimuthal_angle), np.sin(_azimuthal_angle), 0., _rabi_rotation],
        [np.cos(phi_p + _azimuthal_angle), np.sin(phi_p + _azimuthal_angle), 0., np.pi],
        [np.cos(3. * phi_p + _azimuthal_angle),
         np.sin(3. * phi_p + _azimuthal_angle), 0., 2 * np.pi],
        [np.cos(phi_p + _azimuthal_angle), np.sin(phi_p + _azimuthal_angle), 0., np.pi]
    ]

    wimperis_control = new_wimperis_1_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1
    )

    assert np.allclose(wimperis_control.segments, _segments)
