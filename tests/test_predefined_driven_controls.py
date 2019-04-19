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
    PRIMITIVE, WIMPERIS_1, SOLOVAY_KITAEV_1,
    new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control
    #new_compensating_for_off_resonance_with_a_pulse_sequence_control,
    #new_compensating_for_off_resonance_with_a_pulse_sequence_with_solovay_kitaev_control,
    #new_compensating_for_off_resonance_with_a_pulse_sequence_with_wimperis_control,
    #new_walsh_amplitude_modulated_filter_1_control,
    #new_corpse_in_scrofulous_control
)

from qctrlopencontrols.globals import SQUARE


def test_new_predefined_driven_control():
    """Test the new_predefined_driven_control function in
       qctrlopencontrols.driven_controls.predefined
    """
    # Test that an error is raised if supplied with an unknown scheme
    with pytest.raises(ArgumentsValueError):
        _ = new_predefined_driven_control(scheme='nil')

def test_predefined_common_attributes():
    """Test that expected exceptions are raised correctly for invalid parameters
    """
    # Test negative maximum Rabi rate
    with pytest.raises(ArgumentsValueError):
        _ = new_predefined_driven_control(
            maximum_rabi_rate=-1, shape='PRIMITIVE', rabi_rotation=1, azimuthal_angle=0)
    # Test invalid shape
    with pytest.raises(ArgumentsValueError):
        _ = new_predefined_driven_control(
            maximum_rabi_rate=1, shape='-', rabi_rotation=1, azimuthal_angle=0)
    # Test zero Rabi rotation
    with pytest.raises(ArgumentsValueError):
        _ = new_predefined_driven_control(
            maximum_rabi_rate=1, shape='PRIMITIVE', rabi_rotation=0, azimuthal_angle=0)


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

    primitive_control_1 = new_primitive_control(
        rabi_rotation=_rabi_rotation,
        maximum_rabi_rate=_rabi_rate,
        azimuthal_angle=_azimuthal_angle,
        shape=SQUARE
    )

    # Test the new_predefined_driven_control function also
    primitive_control_2 = new_predefined_driven_control(
        rabi_rotation=_rabi_rotation,
        maximum_rabi_rate=_rabi_rate,
        azimuthal_angle=_azimuthal_angle,
        shape=SQUARE,
        scheme=PRIMITIVE
    )

    assert np.allclose(_segments, primitive_control_1.segments)
    assert np.allclose(_segments, primitive_control_2.segments)


def test_wimperis_1_control():
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

    wimperis_control_1 = new_wimperis_1_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1
    )
    wimperis_control_2 = new_predefined_driven_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1,
        scheme=WIMPERIS_1
    )

    assert np.allclose(wimperis_control_1.segments, _segments)
    assert np.allclose(wimperis_control_2.segments, _segments)


def test_solovay_kitaev_1_control():
    """Test the segments of the Solovay-Kitaev 1 (SK1) driven control
    """
    _rabi_rotation = np.pi
    _azimuthal_angle = np.pi/2

    phi_p = np.arccos(-_rabi_rotation / (4 * np.pi))

    _segments = [
        [np.cos(_azimuthal_angle), np.sin(_azimuthal_angle), 0., _rabi_rotation],
        [np.cos(-phi_p + _azimuthal_angle), np.sin(-phi_p + _azimuthal_angle), 0., 2 * np.pi],
        [np.cos(phi_p + _azimuthal_angle), np.sin(phi_p + _azimuthal_angle), 0., 2 * np.pi]
    ]

    sk1_control_1 = new_solovay_kitaev_1_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1
    )

    sk1_control_2 = new_predefined_driven_control(
        scheme=SOLOVAY_KITAEV_1,
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1
    )

    assert np.allclose(sk1_control_1.segments, _segments)
    assert np.allclose(sk1_control_2.segments, _segments)

def test_scofulous_control():
    """Test the segments of the SCROFULOUS driven control.
       Note: here we test against numerical pulse segments since the angles are
       defined numerically as well.
    """
    # Test that exceptions are raised upon wrong inputs for rabi_rotation
    # (SCROFULOUS is only defined for pi/4, pi/2 and pi pulses)
    with pytest.raises(ArgumentsValueError):
        _ = new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control(
            rabi_rotation=0.3
        )

    # Construct SCROFULOUS controls for target rotations pi/4, pi/2 and pi
    pi_segments = new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control(
        rabi_rotation=np.pi, azimuthal_angle=0.5, maximum_rabi_rate=2*np.pi
    ).segments

    _pi_segments = np.array([
        [0.14826172, 6.28143583, 0., 0.5],
        [5.36575214, -3.26911633, 0., 0.5],
        [0.14826172, 6.28143583, 0., 0.5]])

    assert np.allclose(pi_segments, _pi_segments)

    pi_on_2_segments = new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control(
        rabi_rotation=np.pi/2, azimuthal_angle=-0.5, maximum_rabi_rate=2*np.pi
    ).segments

    _pi_on_2_segments = np.array([
        [5.25211762, 3.44872124, 0., 0.32],
        [-1.95046211, -5.97278119, 0., 0.5],
        [5.25211762, 3.44872124, 0., 0.32]])

    assert np.allclose(pi_on_2_segments, _pi_on_2_segments)

    pi_on_4_segments = new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control(
        rabi_rotation=np.pi/4, azimuthal_angle=0, maximum_rabi_rate=2*np.pi
    ).segments

    _pi_on_4_segments = np.array([
        [1.78286387, 6.0249327, 0., 0.26861111],
        [0.54427724, -6.25956707, 0., 0.5],
        [1.78286387, 6.0249327, 0., 0.26861111]])

    assert np.allclose(pi_on_4_segments, _pi_on_4_segments)
