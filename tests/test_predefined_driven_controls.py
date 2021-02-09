# Copyright 2020 Q-CTRL Pty Ltd & Q-CTRL Inc
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
Tests for Predefined Driven Controls.
"""

import numpy as np
import pytest

from qctrlopencontrols.constants import SIGMA_X
from qctrlopencontrols.driven_controls.predefined import (
    new_bb1_control,
    new_corpse_control,
    new_corpse_in_bb1_control,
    new_corpse_in_scrofulous_control,
    new_corpse_in_sk1_control,
    new_modulated_gaussian_control,
    new_primitive_control,
    new_scrofulous_control,
    new_sk1_control,
    new_wamf1_control,
)
from qctrlopencontrols.exceptions import ArgumentsValueError


def test_primitive_control_segments():
    """
    Tests the segments of the predefined primitive driven control.
    """
    _rabi_rate = 1
    _rabi_rotation = 1.5
    _azimuthal_angle = np.pi / 2
    _segments = [
        np.cos(_azimuthal_angle),
        np.sin(_azimuthal_angle),
        0.0,
        _rabi_rotation,
    ]

    primitive_control_1 = new_primitive_control(
        rabi_rotation=_rabi_rotation,
        maximum_rabi_rate=_rabi_rate,
        azimuthal_angle=_azimuthal_angle,
    )

    # Test the new_predefined_driven_control function also
    primitive_control_2 = new_primitive_control(
        rabi_rotation=_rabi_rotation,
        maximum_rabi_rate=_rabi_rate,
        azimuthal_angle=_azimuthal_angle,
    )

    for control in [primitive_control_1, primitive_control_2]:
        segments = [
            control.amplitude_x[0],
            control.amplitude_y[0],
            control.detunings[0],
            control.durations[0],
        ]
        assert np.allclose(_segments, segments)
        assert np.allclose(_rabi_rate, control.maximum_rabi_rate)


def test_wimperis_1_control():
    """
    Tests the segments of the Wimperis 1 (BB1) driven control.
    """
    _rabi_rotation = np.pi
    _azimuthal_angle = np.pi / 2
    _maximum_rabi_rate = 1

    phi_p = np.arccos(-_rabi_rotation / (4 * np.pi))

    _segments = np.array(
        [
            [np.cos(_azimuthal_angle), np.sin(_azimuthal_angle)],
            [np.cos(phi_p + _azimuthal_angle), np.sin(phi_p + _azimuthal_angle)],
            [
                np.cos(3.0 * phi_p + _azimuthal_angle),
                np.sin(3.0 * phi_p + _azimuthal_angle),
            ],
            [np.cos(phi_p + _azimuthal_angle), np.sin(phi_p + _azimuthal_angle)],
        ]
    )

    wimperis_control_1 = new_bb1_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=_maximum_rabi_rate,
    )
    wimperis_control_2 = new_bb1_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=_maximum_rabi_rate,
    )

    durations = [np.pi, np.pi, np.pi * 2, np.pi]

    for control in [wimperis_control_1, wimperis_control_2]:
        segments = np.vstack((control.amplitude_x, control.amplitude_y)).T
        assert np.allclose(segments, _segments)
        assert np.allclose(control.durations, durations)
        assert np.allclose(control.detunings, 0)


def test_solovay_kitaev_1_control():
    """
    Tests the segments of the Solovay-Kitaev 1 (SK1) driven control.
    """
    _rabi_rotation = np.pi
    _azimuthal_angle = np.pi / 2

    phi_p = np.arccos(-_rabi_rotation / (4 * np.pi))

    _segments = [
        [np.cos(_azimuthal_angle), np.sin(_azimuthal_angle)],
        [np.cos(-phi_p + _azimuthal_angle), np.sin(-phi_p + _azimuthal_angle)],
        [np.cos(phi_p + _azimuthal_angle), np.sin(phi_p + _azimuthal_angle)],
    ]

    sk1_control_1 = new_sk1_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1,
    )

    sk1_control_2 = new_sk1_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1,
    )

    durations = [np.pi, 2 * np.pi, 2 * np.pi]

    for control in [sk1_control_1, sk1_control_2]:
        segments = np.vstack((control.amplitude_x, control.amplitude_y)).T
        assert np.allclose(segments, _segments)
        assert np.allclose(control.durations, durations)
        assert np.allclose(control.detunings, 0)


def test_scofulous_control():
    """
    Tests the segments of the SCROFULOUS driven control.

    Note: here we test against numerical pulse segments since the angles are
    defined numerically as well.
    """
    # Test that exceptions are raised upon wrong inputs for rabi_rotation
    # (SCROFULOUS is only defined for pi/4, pi/2 and pi pulses)
    with pytest.raises(ArgumentsValueError):
        _ = new_scrofulous_control(rabi_rotation=0.3, maximum_rabi_rate=2 * np.pi)

    # Construct SCROFULOUS controls for target rotations pi/4, pi/2 and pi
    scrofulous_pi = new_scrofulous_control(
        rabi_rotation=np.pi, azimuthal_angle=0.5, maximum_rabi_rate=2 * np.pi,
    )

    pi_segments = np.vstack(
        (
            scrofulous_pi.amplitude_x,
            scrofulous_pi.amplitude_y,
            scrofulous_pi.detunings,
            scrofulous_pi.durations,
        )
    ).T

    _pi_segments = np.array(
        [
            [0.14826172, 6.28143583, 0.0, 0.5],
            [5.36575214, -3.26911633, 0.0, 0.5],
            [0.14826172, 6.28143583, 0.0, 0.5],
        ]
    )

    assert np.allclose(pi_segments, _pi_segments)

    scrofulous_pi2 = new_scrofulous_control(
        rabi_rotation=np.pi / 2, azimuthal_angle=-0.5, maximum_rabi_rate=2 * np.pi,
    )

    pi_on_2_segments = np.vstack(
        (
            scrofulous_pi2.amplitude_x,
            scrofulous_pi2.amplitude_y,
            scrofulous_pi2.detunings,
            scrofulous_pi2.durations,
        )
    ).T

    _pi_on_2_segments = np.array(
        [
            [5.25211762, 3.44872124, 0.0, 0.32],
            [-1.95046211, -5.97278119, 0.0, 0.5],
            [5.25211762, 3.44872124, 0.0, 0.32],
        ]
    )

    assert np.allclose(pi_on_2_segments, _pi_on_2_segments)

    scrofulous_pi4 = new_scrofulous_control(
        rabi_rotation=np.pi / 4, azimuthal_angle=0, maximum_rabi_rate=2 * np.pi,
    )

    pi_on_4_segments = np.vstack(
        (
            scrofulous_pi4.amplitude_x,
            scrofulous_pi4.amplitude_y,
            scrofulous_pi4.detunings,
            scrofulous_pi4.durations,
        )
    ).T

    _pi_on_4_segments = np.array(
        [
            [1.78286387, 6.0249327, 0.0, 0.26861111],
            [0.54427724, -6.25956707, 0.0, 0.5],
            [1.78286387, 6.0249327, 0.0, 0.26861111],
        ]
    )

    assert np.allclose(pi_on_4_segments, _pi_on_4_segments)


def test_corpse_in_scrofulous_control():
    """
    Test the segments of the CORPSE in SCROFULOUS driven control.

    Note: here we test against numerical pulse segments since the SCROFULOUS angles are
    defined numerically as well.
    """
    # Test pi and pi/2 rotations
    cs_pi = new_corpse_in_scrofulous_control(
        rabi_rotation=np.pi, azimuthal_angle=0.5, maximum_rabi_rate=2 * np.pi,
    )

    pi_segments = np.vstack(
        (cs_pi.amplitude_x, cs_pi.amplitude_y, cs_pi.detunings, cs_pi.durations)
    ).T

    _pi_segments = np.array(
        [
            [0.14826172, 6.28143583, 0.0, 1.16666667],
            [-0.14826172, -6.28143583, 0.0, 0.83333333],
            [0.14826172, 6.28143583, 0.0, 0.16666667],
            [5.36575214, -3.26911633, 0.0, 1.16666667],
            [-5.36575214, 3.26911633, 0.0, 0.83333333],
            [5.36575214, -3.26911633, 0.0, 0.16666667],
            [0.14826172, 6.28143583, 0.0, 1.16666667],
            [-0.14826172, -6.28143583, 0.0, 0.83333333],
            [0.14826172, 6.28143583, 0.0, 0.16666667],
        ]
    )

    assert np.allclose(pi_segments, _pi_segments)

    cs_pi_on_2 = new_corpse_in_scrofulous_control(
        rabi_rotation=np.pi / 2, azimuthal_angle=0.25, maximum_rabi_rate=np.pi,
    )

    pi_on_2_segments = np.vstack(
        (
            cs_pi_on_2.amplitude_x,
            cs_pi_on_2.amplitude_y,
            cs_pi_on_2.detunings,
            cs_pi_on_2.durations,
        )
    ).T

    _pi_on_2_segments = np.array(
        [
            [0.74606697, 3.05171894, 0.0, 2.18127065],
            [-0.74606697, -3.05171894, 0.0, 1.7225413],
            [0.74606697, 3.05171894, 0.0, 0.18127065],
            [1.32207387, -2.84986405, 0.0, 2.33333333],
            [-1.32207387, 2.84986405, 0.0, 1.66666667],
            [1.32207387, -2.84986405, 0.0, 0.33333333],
            [0.74606697, 3.05171894, 0.0, 2.18127065],
            [-0.74606697, -3.05171894, 0.0, 1.7225413],
            [0.74606697, 3.05171894, 0.0, 0.18127065],
        ]
    )

    assert np.allclose(pi_on_2_segments, _pi_on_2_segments)


def test_corpse_control():
    """
    Tests the segments of the CORPSE driven control.
    """
    _rabi_rotation = np.pi
    _azimuthal_angle = np.pi / 4

    k = np.arcsin(np.sin(_rabi_rotation / 2.0) / 2.0)

    _segments = [
        [
            np.cos(_azimuthal_angle),
            np.sin(_azimuthal_angle),
            0.0,
            2.0 * np.pi + _rabi_rotation / 2.0 - k,
        ],
        [
            np.cos(np.pi + _azimuthal_angle),
            np.sin(np.pi + _azimuthal_angle),
            0.0,
            2.0 * np.pi - 2.0 * k,
        ],
        [
            np.cos(_azimuthal_angle),
            np.sin(_azimuthal_angle),
            0.0,
            _rabi_rotation / 2.0 - k,
        ],
    ]

    corpse_control_1 = new_corpse_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1,
    )

    corpse_control_2 = new_corpse_control(
        rabi_rotation=_rabi_rotation,
        azimuthal_angle=_azimuthal_angle,
        maximum_rabi_rate=1,
    )

    for control in [corpse_control_1, corpse_control_2]:
        segments = np.vstack(
            (
                control.amplitude_x,
                control.amplitude_y,
                control.detunings,
                control.durations,
            )
        ).T
        assert np.allclose(segments, _segments)


def test_cinbb_control():
    """
    Tests the segments of the CinBB (BB1 made up of CORPSEs) driven control.
    """
    cinbb = new_corpse_in_bb1_control(
        rabi_rotation=np.pi / 3, azimuthal_angle=0.25, maximum_rabi_rate=np.pi,
    )

    segments = np.vstack(
        (cinbb.amplitude_x, cinbb.amplitude_y, cinbb.detunings, cinbb.durations)
    ).T

    _segments = np.array(
        [
            [3.04392815, 0.77724246, 0.0, 2.08623604],
            [-3.04392815, -0.77724246, 0.0, 1.83913875],
            [3.04392815, 0.77724246, 0.0, 0.08623604],
            [-1.02819968, 2.96857033, 0.0, 1.0],
            [1.50695993, -2.75656964, 0.0, 2.0],
            [-1.02819968, 2.96857033, 0.0, 1.0],
        ]
    )

    assert np.allclose(segments, _segments)

    cinbb = new_corpse_in_bb1_control(
        rabi_rotation=np.pi / 5, azimuthal_angle=-0.25, maximum_rabi_rate=np.pi,
    )

    segments = np.vstack(
        (cinbb.amplitude_x, cinbb.amplitude_y, cinbb.detunings, cinbb.durations)
    ).T

    _segments = np.array(
        [
            [3.04392815, -0.77724246, 0.0, 2.0506206],
            [-3.04392815, 0.77724246, 0.0, 1.9012412],
            [3.04392815, -0.77724246, 0.0, 0.0506206],
            [0.62407389, 3.07898298, 0.0, 1.0],
            [-0.31344034, -3.12591739, 0.0, 2.0],
            [0.62407389, 3.07898298, 0.0, 1.0],
        ]
    )

    assert np.allclose(segments, _segments)


def test_cinsk1_control():
    """
    Tests the segments of the CinSK1 (SK1 made up of CORPSEs) driven control.
    """
    cinsk = new_corpse_in_sk1_control(
        rabi_rotation=np.pi / 2, azimuthal_angle=0.5, maximum_rabi_rate=2 * np.pi,
    )

    segments = np.vstack(
        (cinsk.amplitude_x, cinsk.amplitude_y, cinsk.detunings, cinsk.durations)
    ).T

    _segments = np.array(
        [
            [5.51401386, 3.0123195, 0.0, 1.06748664],
            [-5.51401386, -3.0123195, 0.0, 0.88497327],
            [5.51401386, 3.0123195, 0.0, 0.06748664],
            [2.29944137, -5.84730596, 0.0, 1.0],
            [-3.67794483, 5.09422609, 0.0, 1.0],
        ]
    )

    assert np.allclose(segments, _segments)

    cinsk = new_corpse_in_sk1_control(
        rabi_rotation=2 * np.pi, azimuthal_angle=-0.5, maximum_rabi_rate=2 * np.pi,
    )

    segments = np.vstack(
        (cinsk.amplitude_x, cinsk.amplitude_y, cinsk.detunings, cinsk.durations)
    ).T

    _segments = np.array(
        [
            [5.51401386, -3.0123195, 0.0, 1.5],
            [-5.51401386, 3.0123195, 0.0, 1.0],
            [5.51401386, -3.0123195, 0.0, 0.5],
            [-5.36575214, -3.26911633, 0.0, 1.0],
            [-0.14826172, 6.28143583, 0.0, 1.0],
        ]
    )

    assert np.allclose(segments, _segments)


def test_walsh_control():
    """
    Tests the segments of the first order Walsh driven control.
    """
    # Test that exceptions are raised upon wrong inputs for rabi_rotation
    # (WALSH control is only defined for pi/4, pi/2 and pi pulses)
    with pytest.raises(ArgumentsValueError):
        _ = new_wamf1_control(rabi_rotation=0.3, maximum_rabi_rate=np.pi)

    # test pi rotation
    walsh_pi = new_wamf1_control(
        rabi_rotation=np.pi, azimuthal_angle=-0.35, maximum_rabi_rate=2 * np.pi,
    )

    pi_segments = np.vstack(
        (
            walsh_pi.amplitude_x,
            walsh_pi.amplitude_y,
            walsh_pi.detunings,
            walsh_pi.durations,
        )
    ).T

    _pi_segments = np.array(
        [
            [5.90225283, -2.15449047, 0.0, 0.5],
            [2.95112641, -1.07724523, 0.0, 0.5],
            [2.95112641, -1.07724523, 0.0, 0.5],
            [5.90225283, -2.15449047, 0.0, 0.5],
        ]
    )

    assert np.allclose(pi_segments, _pi_segments)

    # test pi/2 rotation
    walsh_pi_on_2 = new_wamf1_control(
        rabi_rotation=np.pi / 2.0, azimuthal_angle=0.57, maximum_rabi_rate=2 * np.pi,
    )

    pi_on_2_segments = np.vstack(
        (
            walsh_pi_on_2.amplitude_x,
            walsh_pi_on_2.amplitude_y,
            walsh_pi_on_2.detunings,
            walsh_pi_on_2.durations,
        )
    ).T

    _pi_on_2_segments = np.array(
        [
            [5.28981984, 3.39060816, 0.0, 0.39458478],
            [3.08895592, 1.9799236, 0.0, 0.39458478],
            [3.08895592, 1.9799236, 0.0, 0.39458478],
            [5.28981984, 3.39060816, 0.0, 0.39458478],
        ]
    )

    assert np.allclose(pi_on_2_segments, _pi_on_2_segments)

    # test pi/4 rotation
    walsh_pi_on_4 = new_wamf1_control(
        rabi_rotation=np.pi / 4.0, azimuthal_angle=-0.273, maximum_rabi_rate=2 * np.pi,
    )
    pi_on_4_segments = np.vstack(
        (
            walsh_pi_on_4.amplitude_x,
            walsh_pi_on_4.amplitude_y,
            walsh_pi_on_4.detunings,
            walsh_pi_on_4.durations,
        )
    ).T

    _pi_on_4_segments = np.array(
        [
            [6.05049612, -1.69408213, 0.0, 0.3265702],
            [4.37116538, -1.22388528, 0.0, 0.3265702],
            [4.37116538, -1.22388528, 0.0, 0.3265702],
            [6.05049612, -1.69408213, 0.0, 0.3265702],
        ]
    )

    assert np.allclose(pi_on_4_segments, _pi_on_4_segments)


def test_modulated_gaussian_control():
    """
    Tests modulated Gaussian control at different modulate frequencies.
    """
    maximum_rabi_rate = 20 * 2 * np.pi
    minimum_segment_duration = 0.01
    maximum_duration = 0.2

    # set modulation frequency to 0
    pulses_zero = new_modulated_gaussian_control(
        maximum_rabi_rate=maximum_rabi_rate,
        minimum_segment_duration=minimum_segment_duration,
        duration=maximum_duration,
        modulation_frequency=0,
    )
    pulse_zero_segments = [
        {"duration": d, "value": np.real(v)}
        for d, v in zip(
            pulses_zero.durations,
            pulses_zero.rabi_rates * np.exp(1j * pulses_zero.azimuthal_angles),
        )
    ]

    # set modulation frequency to 50/3
    pulses_non_zero = new_modulated_gaussian_control(
        maximum_rabi_rate=maximum_rabi_rate,
        minimum_segment_duration=minimum_segment_duration,
        duration=maximum_duration,
        modulation_frequency=50 / 3,
    )
    pulse_non_zero_segments = [
        {"duration": d, "value": np.real(v)}
        for d, v in zip(
            pulses_non_zero.durations,
            pulses_non_zero.rabi_rates * np.exp(1j * pulses_non_zero.azimuthal_angles),
        )
    ]

    # pulses should have 20 segments; 0.2/0.01 = 20
    assert len(pulse_zero_segments) == 20
    assert len(pulse_non_zero_segments) == 20

    # determine the segment mid-points
    segment_mid_points = 0.2 / 20 * (0.5 + np.arange(20))
    base_gaussian_mean = maximum_duration * 0.5
    base_gaussian_width = maximum_duration * 0.1
    base_gaussian = np.exp(
        -0.5 * ((segment_mid_points - base_gaussian_mean) / base_gaussian_width) ** 2.0
    ) / (np.sqrt(2 * np.pi) * base_gaussian_width)

    # for modulation at frequency = 0
    segment_values = np.array([p["value"] for p in pulse_zero_segments])
    segment_durations = np.array([p["duration"] for p in pulse_zero_segments])
    # The base Gaussian creates a rotation of 1rad and has maximum value
    # 1/(sqrt(2pi)*0.2*0.1)=~19.9. Therefore, with a maximum Rabi rate of 20*2pi, we can achieve
    # only a single 2pi rotation, which corresponds to scaling up the Gaussian by 2pi.
    expected_gaussian = 2 * np.pi * base_gaussian

    assert np.allclose(segment_values, expected_gaussian)
    assert np.allclose(segment_durations, 0.2 / 20)

    # for modulation at frequency = 50/3
    segment_values = np.array([segment["value"] for segment in pulse_non_zero_segments])
    segment_durations = np.array(
        [segment["duration"] for segment in pulse_non_zero_segments]
    )
    expected_base_segments = base_gaussian * np.sin(
        2 * np.pi * (50 / 3) * (segment_mid_points - 0.2 / 2)
    )
    expected_base_segments /= np.max(expected_base_segments)
    expected_base_segments *= maximum_rabi_rate

    assert np.allclose(segment_values, expected_base_segments)
    assert np.allclose(segment_durations, 0.2 / 20)


def test_modulated_gaussian_control_give_identity_gate():
    """
    Tests that the modulated Gaussian sequences produce identity gates.

    Apply the modulated sequences to drive a noiseless qubit rotating along X. The net
    effect should be an identity gate.
    """

    maximum_rabi_rate = 50 * 2 * np.pi
    minimum_segment_duration = 0.02
    maximum_duration = 0.2

    pulses = [
        new_modulated_gaussian_control(
            maximum_rabi_rate=maximum_rabi_rate,
            minimum_segment_duration=minimum_segment_duration,
            duration=maximum_duration,
            modulation_frequency=f,
        )
        for f in [0, 20]
    ]

    unitaries = [
        np.linalg.multi_dot(
            [
                np.cos(d * v) * np.eye(2) + 1j * np.sin(d * v) * SIGMA_X
                for d, v in zip(
                    p.durations, p.rabi_rates * np.exp(1j * p.azimuthal_angles),
                )
            ]
        )
        for p in pulses
    ]

    for _u in unitaries:
        assert np.allclose(_u, np.eye(2))
