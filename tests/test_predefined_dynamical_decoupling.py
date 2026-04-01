# Copyright 2026 Q-CTRL
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
Tests for Predefined DDS.
"""

import numpy as np

from qctrlopencontrols import (
    new_carr_purcell_sequence,
    new_cpmg_sequence,
    new_periodic_sequence,
    new_quadratic_sequence,
    new_ramsey_sequence,
    new_spin_echo_sequence,
    new_uhrig_sequence,
    new_walsh_sequence,
    new_x_concatenated_sequence,
    new_xy_concatenated_sequence,
    new_platonic_sequence,
)
from qctrlopencontrols.constants import (
    SIGMA_X,
    SIGMA_Y,
    SIGMA_Z,
)


def test_ramsey_sequence():
    """
    Tests the Ramsey sequence.
    """

    duration = 10.0

    sequence = new_ramsey_sequence(duration=duration)

    _offsets = np.array([])
    _rabi_rotations = np.array([])
    _azimuthal_angles = np.array([])
    _detuning_rotations = np.array([])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_ramsey_sequence(duration=duration, pre_post_rotation=True)

    _rabi_rotations = np.array([np.pi / 2, np.pi / 2])
    _azimuthal_angles = np.array([0.0, np.pi])
    _detuning_rotations = np.array([0.0, 0.0])

    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_spin_echo_sequence():
    """
    Test the spin echo sequence.
    """

    duration = 10.0

    sequence = new_spin_echo_sequence(duration=duration)

    _offsets = np.array([duration / 2.0])
    _rabi_rotations = np.array([np.pi])
    _azimuthal_angles = np.array([0])
    _detuning_rotations = np.array([0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_spin_echo_sequence(duration=duration, pre_post_rotation=True)

    _offsets = np.array([0, duration / 2.0, duration])
    _rabi_rotations = np.array([np.pi / 2, np.pi, np.pi / 2])
    _azimuthal_angles = np.array([0, 0, 0])
    _detuning_rotations = np.array([0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_carr_purcell_sequence():
    """
    Test the Carr-Purcell sequence.
    """

    duration = 10.0
    offset_count = 4

    sequence = new_carr_purcell_sequence(duration=duration, offset_count=offset_count)

    _spacing = duration / offset_count
    _offsets = np.array(
        [
            _spacing * 0.5,
            _spacing * 0.5 + _spacing,
            _spacing * 0.5 + 2 * _spacing,
            _spacing * 0.5 + 3 * _spacing,
        ]
    )
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([0, 0, 0, 0])
    _detuning_rotations = np.array([0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_carr_purcell_sequence(
        duration=duration, offset_count=offset_count, pre_post_rotation=True
    )

    _offsets = np.array(
        [
            0,
            _spacing * 0.5,
            _spacing * 0.5 + _spacing,
            _spacing * 0.5 + 2 * _spacing,
            _spacing * 0.5 + 3 * _spacing,
            duration,
        ]
    )
    _rabi_rotations = np.array([np.pi / 2, np.pi, np.pi, np.pi, np.pi, np.pi / 2])
    _azimuthal_angles = np.array([0, 0, 0, 0, 0, np.pi])
    _detuning_rotations = np.array([0, 0, 0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_cpmg_sequence():
    """
    Tests the CPMG sequence.
    """

    duration = 10.0
    offset_count = 4

    sequence = new_cpmg_sequence(duration=duration, offset_count=offset_count)

    _spacing = duration / offset_count
    _offsets = np.array(
        [
            _spacing * 0.5,
            _spacing * 0.5 + _spacing,
            _spacing * 0.5 + 2 * _spacing,
            _spacing * 0.5 + 3 * _spacing,
        ]
    )
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
    _detuning_rotations = np.array([0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_cpmg_sequence(
        duration=duration, offset_count=offset_count, pre_post_rotation=True
    )

    _offsets = np.array(
        [
            0,
            _spacing * 0.5,
            _spacing * 0.5 + _spacing,
            _spacing * 0.5 + 2 * _spacing,
            _spacing * 0.5 + 3 * _spacing,
            duration,
        ]
    )
    _rabi_rotations = np.array([np.pi / 2, np.pi, np.pi, np.pi, np.pi, np.pi / 2])
    _azimuthal_angles = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi])
    _detuning_rotations = np.array([0, 0, 0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_uhrig_sequence():
    """
    Tests the Uhrig sequence.
    """

    duration = 10.0
    offset_count = 4

    sequence = new_uhrig_sequence(duration=duration, offset_count=offset_count)

    constant = 0.5 / (offset_count + 1)
    _delta_positions = [
        duration * (np.sin(np.pi * (k + 1) * constant)) ** 2
        for k in range(offset_count)
    ]

    _offsets = np.array(_delta_positions)
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
    _detuning_rotations = np.array([0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_uhrig_sequence(
        duration=duration, offset_count=offset_count, pre_post_rotation=True
    )

    _offsets = np.array(_delta_positions)
    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])

    _rabi_rotations = np.array([np.pi / 2, np.pi, np.pi, np.pi, np.pi, np.pi / 2])
    _azimuthal_angles = np.array(
        [0.0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi]
    )
    _detuning_rotations = np.array([0.0, 0, 0, 0, 0, 0.0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_periodic_sequence():
    """
    Tests the periodic sequence.
    """

    duration = 10.0
    offset_count = 4

    sequence = new_periodic_sequence(duration=duration, offset_count=offset_count)

    constant = 1 / (offset_count + 1)
    # prepare the offsets for delta comb
    _delta_positions = [duration * k * constant for k in range(1, offset_count + 1)]
    _offsets = np.array(_delta_positions)
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([0, 0, 0, 0])
    _detuning_rotations = np.array([0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_periodic_sequence(
        duration=duration, offset_count=offset_count, pre_post_rotation=True
    )

    _offsets = np.array(_delta_positions)
    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])

    _rabi_rotations = np.array([np.pi / 2, np.pi, np.pi, np.pi, np.pi, np.pi / 2])
    _azimuthal_angles = np.array([0, 0, 0, 0, 0, np.pi])
    _detuning_rotations = np.array([0, 0, 0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_walsh_sequence():
    """
    Tests the Walsh sequence.
    """

    duration = 10.0
    paley_order = 20

    sequence = new_walsh_sequence(duration=duration, paley_order=paley_order)

    hamming_weight = 5
    samples = 2**hamming_weight
    relative_offset = np.arange(1.0 / (2 * samples), 1.0, 1.0 / samples)
    binary_string = np.binary_repr(paley_order)
    binary_order = [int(binary_string[i]) for i in range(hamming_weight)]
    walsh_array = np.ones([samples])
    for i in range(hamming_weight):
        walsh_array *= (
            np.sign(np.sin(2 ** (i + 1) * np.pi * relative_offset))
            ** binary_order[hamming_weight - 1 - i]
        )

    walsh_relative_offsets = np.array(
        [
            (i + 1) * (1.0 / samples)
            for i in range(samples - 1)
            if walsh_array[i] != walsh_array[i + 1]
        ]
    )
    _offsets = duration * walsh_relative_offsets

    _rabi_rotations = np.pi * np.ones(_offsets.shape)
    _azimuthal_angles = np.zeros(_offsets.shape)
    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_walsh_sequence(
        duration=duration, paley_order=paley_order, pre_post_rotation=True
    )

    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations, [0, _rabi_rotations.shape[0]], [np.pi / 2, np.pi / 2]
    )
    _azimuthal_angles = np.zeros(_offsets.shape)
    _azimuthal_angles[-1] = np.pi
    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_quadratic_sequence():
    """
    Tests the quadratic sequence.
    """

    duration = 10.0
    inner_offset_count = 4
    outer_offset_count = 4

    sequence = new_quadratic_sequence(
        duration=duration,
        inner_offset_count=inner_offset_count,
        outer_offset_count=outer_offset_count,
    )

    _offsets = np.zeros((outer_offset_count + 1, inner_offset_count + 1))

    constant = 0.5 / (outer_offset_count + 1)
    _outer_offsets = np.array(
        [
            duration * (np.sin(np.pi * (k + 1) * constant)) ** 2
            for k in range(outer_offset_count)
        ]
    )
    _offsets[0:outer_offset_count, -1] = _outer_offsets

    _outer_offsets = np.insert(
        _outer_offsets, [0, _outer_offsets.shape[0]], [0, duration]
    )
    _inner_durations = _outer_offsets[1:] - _outer_offsets[0:-1]

    constant = 0.5 / (inner_offset_count + 1)
    _delta_positions = np.array(
        [(np.sin(np.pi * (k + 1) * constant)) ** 2 for k in range(inner_offset_count)]
    )
    for inner_sequence_idx in range(_inner_durations.shape[0]):
        _inner_deltas = _inner_durations[inner_sequence_idx] * _delta_positions
        _inner_deltas = _outer_offsets[inner_sequence_idx] + _inner_deltas
        _offsets[inner_sequence_idx, 0:inner_offset_count] = _inner_deltas

    _rabi_rotations = np.zeros(_offsets.shape)
    _detuning_rotations = np.zeros(_offsets.shape)

    _rabi_rotations[0:outer_offset_count, -1] = np.pi
    _detuning_rotations[0 : (outer_offset_count + 1), 0:inner_offset_count] = np.pi

    _reshaped_offsets = np.reshape(_offsets, (-1,))[0:-1]
    _reshaped_rabi_rotations = np.reshape(_rabi_rotations, (-1,))[0:-1]
    _reshaped_detuning_rotations = np.reshape(_detuning_rotations, (-1,))[0:-1]
    _azimuthal_angles = np.zeros(_reshaped_offsets.shape)

    assert np.allclose(_reshaped_offsets, sequence.offsets)
    assert np.allclose(_reshaped_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_reshaped_detuning_rotations, sequence.detuning_rotations)

    sequence = new_quadratic_sequence(
        duration=duration,
        inner_offset_count=inner_offset_count,
        outer_offset_count=outer_offset_count,
        pre_post_rotation=True,
    )

    _reshaped_offsets = np.insert(
        _reshaped_offsets, [0, _reshaped_offsets.shape[0]], [0, duration]
    )
    _reshaped_rabi_rotations = np.insert(
        _reshaped_rabi_rotations,
        [0, _reshaped_rabi_rotations.shape[0]],
        [np.pi / 2, np.pi / 2],
    )
    _reshaped_detuning_rotations = np.insert(
        _reshaped_detuning_rotations, [0, _reshaped_detuning_rotations.shape[0]], [0, 0]
    )

    _azimuthal_angles = np.zeros(_reshaped_offsets.shape)
    _azimuthal_angles[-1] = np.pi

    assert np.allclose(_reshaped_offsets, sequence.offsets)
    assert np.allclose(_reshaped_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_reshaped_detuning_rotations, sequence.detuning_rotations)


def test_x_concatenated_sequence():
    """
    Tests the X-concatenated sequence.
    """
    duration = 10.0
    concatenation_order = 3

    sequence = new_x_concatenated_sequence(
        duration=duration, concatenation_order=concatenation_order
    )

    _spacing = duration / (2**concatenation_order)
    _offsets = np.array(
        [_spacing, 3 * _spacing, 4 * _spacing, 5 * _spacing, 7 * _spacing]
    )
    _rabi_rotations = np.pi * np.ones(_offsets.shape)

    _azimuthal_angles = np.zeros(_offsets.shape)
    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_x_concatenated_sequence(
        duration=duration,
        concatenation_order=concatenation_order,
        pre_post_rotation=True,
    )

    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations, [0, _rabi_rotations.shape[0]], [np.pi / 2, np.pi / 2]
    )
    _azimuthal_angles = np.zeros(_offsets.shape)
    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_xy_concatenated_sequence():
    """
    Tests the XY-concatenated sequence.
    """

    duration = 10.0
    concatenation_order = 2

    sequence = new_xy_concatenated_sequence(
        duration=duration, concatenation_order=concatenation_order
    )

    _spacing = duration / (2 ** (concatenation_order * 2))
    _offsets = np.array(
        [
            _spacing,
            2 * _spacing,
            3 * _spacing,
            4 * _spacing,
            5 * _spacing,
            6 * _spacing,
            7 * _spacing,
            9 * _spacing,
            10 * _spacing,
            11 * _spacing,
            12 * _spacing,
            13 * _spacing,
            14 * _spacing,
            15 * _spacing,
        ]
    )
    _rabi_rotations = np.array(
        [
            np.pi,
            np.pi,
            np.pi,
            0.0,
            np.pi,
            np.pi,
            np.pi,
            np.pi,
            np.pi,
            np.pi,
            0,
            np.pi,
            np.pi,
            np.pi,
        ]
    )
    _azimuthal_angles = np.array(
        [0, np.pi / 2, 0, 0, 0, np.pi / 2, 0, 0, np.pi / 2, 0, 0, 0, np.pi / 2, 0]
    )
    _detuning_rotations = np.array([0, 0, 0, np.pi, 0, 0, 0, 0, 0, 0, np.pi, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_xy_concatenated_sequence(
        duration=duration,
        concatenation_order=concatenation_order,
        pre_post_rotation=True,
    )

    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations, [0, _rabi_rotations.shape[0]], [np.pi / 2, np.pi / 2]
    )
    _azimuthal_angles = np.insert(
        _azimuthal_angles, [0, _azimuthal_angles.shape[0]], [0, np.pi]
    )
    _detuning_rotations = np.insert(
        _detuning_rotations, [0, _detuning_rotations.shape[0]], [0, 0]
    )

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def _pulses_produce_identity(sequence):
    """
    Tests if the pulses of a DDS sequence produce an identity or Z rotation in absence of noise.
    We check this by creating the unitary of each pulse and then multiplying them
    by each other to check the complete evolution.
    """

    # The unitary evolution due to an instantaneous pulse can be written as
    # U = cos(|n|) I -i sin(|n|) *(n_x SIGMA_x + n_y SIGMA_y + n_z SIGMA_z)/|n|
    # where n is a vector with components
    # n_x = rabi * cos(azimuthal)/2
    # n_y = rabi * sin(azimuthal)/2
    # n_z = detuning/2

    matrix_product = np.identity(2)
    for rabi, azimuth, detuning in zip(
        sequence.rabi_rotations, sequence.azimuthal_angles, sequence.detuning_rotations
    ):
        n_x = rabi * np.cos(azimuth) / 2.0
        n_y = rabi * np.sin(azimuth) / 2.0
        n_z = detuning / 2.0
        mod_n = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        unitary = (
            np.cos(mod_n) * np.identity(2)
            - 1.0j * (np.sin(mod_n) * n_x / mod_n) * SIGMA_X
            - 1.0j * (np.sin(mod_n) * n_y / mod_n) * SIGMA_Y
            - 1.0j * (np.sin(mod_n) * n_z / mod_n) * SIGMA_Z
        )
        matrix_product = np.matmul(unitary, matrix_product)

    # Remove global phase
    matrix_product *= np.exp(-1.0j * np.angle(matrix_product[0][0]))

    expected_matrix_product = np.identity(2)

    return np.allclose(matrix_product, expected_matrix_product) or np.allclose(
        SIGMA_Z.dot(matrix_product), expected_matrix_product
    )


def test_if_ramsey_sequence_is_identity():
    """
    Tests if the product of the pulses in the Ramsey sequence with pre/post
    pi/2-pulses is an identity.
    """
    ramsey_sequence = new_ramsey_sequence(duration=10.0, pre_post_rotation=True)

    assert _pulses_produce_identity(ramsey_sequence)


def test_if_spin_echo_sequence_is_identity():
    """
    Tests if the product of the pulses in a Spin Echo sequence with pre/post
    pi/2-pulses is an identity.
    """
    spin_echo_sequence = new_spin_echo_sequence(duration=10.0, pre_post_rotation=True)

    assert _pulses_produce_identity(spin_echo_sequence)


def test_if_carr_purcell_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a Carr-Purcell sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is odd.
    """
    odd_carr_purcell_sequence = new_carr_purcell_sequence(
        duration=10.0, offset_count=7, pre_post_rotation=True
    )

    assert _pulses_produce_identity(odd_carr_purcell_sequence)


def test_if_carr_purcell_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a Carr-Purcell sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_carr_purcell_sequence = new_carr_purcell_sequence(
        duration=10.0, offset_count=8, pre_post_rotation=True
    )

    assert _pulses_produce_identity(even_carr_purcell_sequence)


def test_if_cpmg_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a CPMG sequence with pre/post
    pi/2-pulses and an extra Z rotation is an identity, when the number of pulses is odd.
    """
    odd_cpmg_sequence = new_cpmg_sequence(
        duration=10.0, offset_count=7, pre_post_rotation=True
    )

    assert _pulses_produce_identity(odd_cpmg_sequence)


def test_if_cpmg_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a CPMG sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_cpmg_sequence = new_cpmg_sequence(
        duration=10.0, offset_count=8, pre_post_rotation=True
    )

    assert _pulses_produce_identity(even_cpmg_sequence)


def test_if_uhrig_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in an Uhrig sequence with pre/post
    pi/2-pulses and an extra Z rotation is an identity, when the number of pulses is odd.
    """
    odd_uhrig_sequence = new_uhrig_sequence(
        duration=10.0, offset_count=7, pre_post_rotation=True
    )

    assert _pulses_produce_identity(odd_uhrig_sequence)


def test_if_uhrig_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in an Uhrig sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_uhrig_sequence = new_uhrig_sequence(
        duration=10.0, offset_count=8, pre_post_rotation=True
    )

    assert _pulses_produce_identity(even_uhrig_sequence)


def test_if_periodic_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a periodic DDS with pre/post
    pi/2-pulses is an identity, when the number of pulses is odd.
    """
    odd_periodic_sequence = new_periodic_sequence(
        duration=10.0, offset_count=7, pre_post_rotation=True
    )

    assert _pulses_produce_identity(odd_periodic_sequence)


def test_if_periodic_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a periodic DDS with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_periodic_sequence = new_periodic_sequence(
        duration=10.0, offset_count=8, pre_post_rotation=True
    )

    assert _pulses_produce_identity(even_periodic_sequence)


def test_if_walsh_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a Walsh sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is odd.
    """
    odd_walsh_sequence = new_walsh_sequence(
        duration=10.0, paley_order=7, pre_post_rotation=True
    )

    # A Walsh sequence with paley_order 7 has 5 pi-pulses + 2 pi/2-pulses,
    # see https://arxiv.org/pdf/1109.6002.pdf
    assert len(odd_walsh_sequence.offsets) == 5 + 2

    assert _pulses_produce_identity(odd_walsh_sequence)


def test_if_walsh_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_walsh_sequence = new_walsh_sequence(
        duration=10.0, paley_order=6, pre_post_rotation=True
    )

    # A Walsh sequence with paley_order 6 has 4 pi-pulses + 2 pi/2-pulses,
    # see https://arxiv.org/pdf/1109.6002.pdf
    assert len(even_walsh_sequence.offsets) == 4 + 2

    assert _pulses_produce_identity(even_walsh_sequence)


def test_if_quadratic_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the total number of pulses is odd.
    """
    odd_quadratic_sequence = new_quadratic_sequence(
        duration=10.0,
        inner_offset_count=7,
        outer_offset_count=7,
        pre_post_rotation=True,
    )

    # n_outer + n_inner*(n_outer+1) pi-pulses + 2 pi/2-pulses
    # total number here is odd
    assert len(odd_quadratic_sequence.offsets) == 7 + 7 * (7 + 1) + 2

    assert _pulses_produce_identity(odd_quadratic_sequence)


def test_if_quadratic_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the total number of pulses is even.
    """
    even_quadratic_sequence = new_quadratic_sequence(
        duration=10.0,
        inner_offset_count=8,
        outer_offset_count=8,
        pre_post_rotation=True,
    )

    # n_outer + n_inner*(n_outer+1) pi-pulses + 2 pi/2-pulses
    # total number here is even
    assert len(even_quadratic_sequence.offsets) == 8 + 8 * (8 + 1) + 2

    assert _pulses_produce_identity(even_quadratic_sequence)


def test_if_quadratic_sequence_with_odd_inner_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses and an extra rotation is an identity, when the total number
    of inner pulses is odd.
    """
    inner_odd_quadratic_sequence = new_quadratic_sequence(
        duration=10.0,
        inner_offset_count=7,
        outer_offset_count=8,
        pre_post_rotation=True,
    )

    # n_outer + n_inner*(n_outer+1) pi-pulses + 2 pi/2-pulses
    # total number here is odd
    assert len(inner_odd_quadratic_sequence.offsets) == 8 + 7 * (8 + 1) + 2

    assert _pulses_produce_identity(inner_odd_quadratic_sequence)


def test_if_quadratic_sequence_with_even_inner_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the total number of inner pulses is even.
    """
    inner_even_quadratic_sequence = new_quadratic_sequence(
        duration=10.0,
        inner_offset_count=8,
        outer_offset_count=7,
        pre_post_rotation=True,
    )

    # n_outer + n_inner*(n_outer+1) pi-pulses + 2 pi/2-pulses
    # total number here is even
    assert len(inner_even_quadratic_sequence.offsets) == 7 + 8 * (7 + 1) + 2

    assert _pulses_produce_identity(inner_even_quadratic_sequence)


def test_if_x_concatenated_sequence_is_identity():
    """
    Tests if the product of the pulses in an X concatenated sequence with pre/post
    pi/2-pulses is an identity.
    """
    x_concat_sequence = new_x_concatenated_sequence(
        duration=10.0, concatenation_order=4, pre_post_rotation=True
    )

    assert _pulses_produce_identity(x_concat_sequence)


def test_if_xy_concatenated_sequence_is_identity():
    """
    Tests if the product of the pulses in an XY concatenated sequence with pre/post
    pi/2-pulses is an identity.
    """
    xy_concat_sequence = new_xy_concatenated_sequence(
        duration=10.0, concatenation_order=4, pre_post_rotation=True
    )

    assert _pulses_produce_identity(xy_concat_sequence)


def test_dihedral_platonic_sequence():
    """
    Tests the Dihedral order of the platonic sequence.
    """
    duration = 10.0
    sequence = new_platonic_sequence(duration=duration, sequence="Dihedral")
    count = 8

    _spacing = duration / count

    _offsets = np.array(
        [
            _spacing * 0.5,
            _spacing * 0.5 + _spacing,
            _spacing * 0.5 + 2 * _spacing,
            _spacing * 0.5 + 3 * _spacing,
            _spacing * 0.5 + 4 * _spacing,
            _spacing * 0.5 + 5 * _spacing,
            _spacing * 0.5 + 6 * _spacing,
            _spacing * 0.5 + 7 * _spacing,
        ]
    )

    _rabi_rotations = np.ones(_offsets.shape) * np.pi

    _azimuthal_angles = np.array(
        [0, np.pi / 2, 0, np.pi / 2, np.pi / 2, 0, np.pi / 2, 0]
    )

    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_platonic_sequence(
        duration=duration, sequence="Dihedral", pre_post_rotation=True
    )

    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations, [0, _rabi_rotations.shape[0]], [np.pi / 2, np.pi / 2]
    )
    _azimuthal_angles = np.insert(
        _azimuthal_angles, [0, _azimuthal_angles.shape[0]], [0, np.pi]
    )
    _detuning_rotations = np.insert(
        _detuning_rotations, [0, _detuning_rotations.shape[0]], [0, 0]
    )

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_tetrahedral_platonic_sequence():
    """
    Tests the Tetrahedral order of the platonic sequence.
    """
    duration = 10.0
    sequence = new_platonic_sequence(duration=duration, sequence="Tetrahedral")

    count = 24
    _spacing = duration / count

    _offsets = np.array(
        [
            _spacing * 0.5,
            _spacing * 0.5 + _spacing,
            _spacing * 0.5 + 2 * _spacing,
            _spacing * 0.5 + 3 * _spacing,
            _spacing * 0.5 + 4 * _spacing,
            _spacing * 0.5 + 5 * _spacing,
            _spacing * 0.5 + 6 * _spacing,
            _spacing * 0.5 + 7 * _spacing,
            _spacing * 0.5 + 8 * _spacing,
            _spacing * 0.5 + 9 * _spacing,
            _spacing * 0.5 + 10 * _spacing,
            _spacing * 0.5 + 11 * _spacing,
            _spacing * 0.5 + 12 * _spacing,
            _spacing * 0.5 + 13 * _spacing,
            _spacing * 0.5 + 14 * _spacing,
            _spacing * 0.5 + 15 * _spacing,
            _spacing * 0.5 + 16 * _spacing,
            _spacing * 0.5 + 17 * _spacing,
            _spacing * 0.5 + 18 * _spacing,
            _spacing * 0.5 + 19 * _spacing,
            _spacing * 0.5 + 20 * _spacing,
            _spacing * 0.5 + 21 * _spacing,
            _spacing * 0.5 + 22 * _spacing,
            _spacing * 0.5 + 23 * _spacing,
        ]
    )

    _rabi_rotations = np.array(
        [
            0,
            4 * np.sqrt(2) * np.pi / 9,
            0,
            0,
            4 * np.sqrt(2) * np.pi / 9,
            0,
            4 * np.sqrt(2) * np.pi / 9,
            4 * np.sqrt(2) * np.pi / 9,
            4 * np.sqrt(2) * np.pi / 9,
            0,
            0,
            4 * np.sqrt(2) * np.pi / 9,
            0,
            4 * np.sqrt(2) * np.pi / 9,
            4 * np.sqrt(2) * np.pi / 9,
            4 * np.sqrt(2) * np.pi / 9,
            0,
            0,
            4 * np.sqrt(2) * np.pi / 9,
            0,
            4 * np.sqrt(2) * np.pi / 9,
            4 * np.sqrt(2) * np.pi / 9,
            0,
            0,
        ]
    )

    _azimuthal_angles = np.array(
        [
            0,
            np.pi / 3,
            0,
            0,
            np.pi / 3,
            0,
            np.pi / 3,
            np.pi / 3,
            np.pi / 3,
            0,
            0,
            np.pi / 3,
            0,
            np.pi / 3,
            np.pi / 3,
            np.pi / 3,
            0,
            0,
            np.pi / 3,
            0,
            np.pi / 3,
            np.pi / 3,
            0,
            0,
        ]
    )

    _detuning_rotations = np.array(
        [
            2 * np.pi / 3,
            2 * np.pi / 9,
            2 * np.pi / 3,
            2 * np.pi / 3,
            2 * np.pi / 9,
            2 * np.pi / 3,
            2 * np.pi / 9,
            2 * np.pi / 9,
            2 * np.pi / 9,
            2 * np.pi / 3,
            2 * np.pi / 3,
            2 * np.pi / 9,
            2 * np.pi / 3,
            2 * np.pi / 9,
            2 * np.pi / 9,
            2 * np.pi / 9,
            2 * np.pi / 3,
            2 * np.pi / 3,
            2 * np.pi / 9,
            2 * np.pi / 3,
            2 * np.pi / 9,
            2 * np.pi / 9,
            2 * np.pi / 3,
            2 * np.pi / 3,
        ]
    )

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_platonic_sequence(
        duration=duration, sequence="Tetrahedral", pre_post_rotation=True
    )

    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations, [0, _rabi_rotations.shape[0]], [np.pi / 2, np.pi / 2]
    )
    _azimuthal_angles = np.insert(
        _azimuthal_angles, [0, _azimuthal_angles.shape[0]], [0, np.pi]
    )
    _detuning_rotations = np.insert(
        _detuning_rotations, [0, _detuning_rotations.shape[0]], [0, 0]
    )

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_octahedral_platonic_sequence():
    """
    Tests the Octahedral order of the platonic sequence.
    """
    duration = 10.0
    sequence = new_platonic_sequence(duration=duration, sequence="Octahedral")

    count = 48
    _spacing = duration / count

    _offsets = np.array(
        [
            _spacing * 0.5,
            _spacing * 0.5 + _spacing,
            _spacing * 0.5 + 2 * _spacing,
            _spacing * 0.5 + 3 * _spacing,
            _spacing * 0.5 + 4 * _spacing,
            _spacing * 0.5 + 5 * _spacing,
            _spacing * 0.5 + 6 * _spacing,
            _spacing * 0.5 + 7 * _spacing,
            _spacing * 0.5 + 8 * _spacing,
            _spacing * 0.5 + 9 * _spacing,
            _spacing * 0.5 + 10 * _spacing,
            _spacing * 0.5 + 11 * _spacing,
            _spacing * 0.5 + 12 * _spacing,
            _spacing * 0.5 + 13 * _spacing,
            _spacing * 0.5 + 14 * _spacing,
            _spacing * 0.5 + 15 * _spacing,
            _spacing * 0.5 + 16 * _spacing,
            _spacing * 0.5 + 17 * _spacing,
            _spacing * 0.5 + 18 * _spacing,
            _spacing * 0.5 + 19 * _spacing,
            _spacing * 0.5 + 20 * _spacing,
            _spacing * 0.5 + 21 * _spacing,
            _spacing * 0.5 + 22 * _spacing,
            _spacing * 0.5 + 23 * _spacing,
            _spacing * 0.5 + 24 * _spacing,
            _spacing * 0.5 + 25 * _spacing,
            _spacing * 0.5 + 26 * _spacing,
            _spacing * 0.5 + 27 * _spacing,
            _spacing * 0.5 + 28 * _spacing,
            _spacing * 0.5 + 29 * _spacing,
            _spacing * 0.5 + 30 * _spacing,
            _spacing * 0.5 + 31 * _spacing,
            _spacing * 0.5 + 32 * _spacing,
            _spacing * 0.5 + 33 * _spacing,
            _spacing * 0.5 + 34 * _spacing,
            _spacing * 0.5 + 35 * _spacing,
            _spacing * 0.5 + 36 * _spacing,
            _spacing * 0.5 + 37 * _spacing,
            _spacing * 0.5 + 38 * _spacing,
            _spacing * 0.5 + 39 * _spacing,
            _spacing * 0.5 + 40 * _spacing,
            _spacing * 0.5 + 41 * _spacing,
            _spacing * 0.5 + 42 * _spacing,
            _spacing * 0.5 + 43 * _spacing,
            _spacing * 0.5 + 44 * _spacing,
            _spacing * 0.5 + 45 * _spacing,
            _spacing * 0.5 + 46 * _spacing,
            _spacing * 0.5 + 47 * _spacing,
        ]
    )

    _rabi_rotations = np.array(
        [
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            0,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            0,
            0,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            0,
            0,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            0,
            2 * np.sqrt(2 / 3) * np.pi / 3,
            2 * np.sqrt(2 / 3) * np.pi / 3,
        ]
    )

    _azimuthal_angles = np.array(
        [
            0,
            np.pi / 4,
            0,
            0,
            0,
            np.pi / 4,
            np.pi / 4,
            np.pi / 4,
            0,
            np.pi / 4,
            0,
            0,
            np.pi / 4,
            np.pi / 4,
            np.pi / 4,
            0,
            0,
            np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 4,
            0,
            0,
            0,
            0,
            np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 4,
            np.pi / 4,
            0,
            np.pi / 4,
            0,
            0,
            np.pi / 4,
            np.pi / 4,
            0,
            0,
            0,
            0,
            np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 4,
            np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 4,
        ]
    )

    _detuning_rotations = np.array(
        [
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
            np.pi / 2,
            2 * np.pi / 3 / np.sqrt(3),
            2 * np.pi / 3 / np.sqrt(3),
        ]
    )

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_platonic_sequence(
        duration=duration, sequence="Octahedral", pre_post_rotation=True
    )

    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations, [0, _rabi_rotations.shape[0]], [np.pi / 2, np.pi / 2]
    )
    _azimuthal_angles = np.insert(
        _azimuthal_angles, [0, _azimuthal_angles.shape[0]], [0, np.pi]
    )
    _detuning_rotations = np.insert(
        _detuning_rotations, [0, _detuning_rotations.shape[0]], [0, 0]
    )

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_icosahedral_platonic_sequence():
    """
    Tests the Icosahedral order of the platonic sequence.
    """
    duration = 10.0

    sequence = new_platonic_sequence(duration=duration, sequence="Icosahedral")

    count = 120
    _spacing = duration / count

    _offsets = np.array(
        [
            _spacing * 0.5,
            _spacing * 0.5 + _spacing,
            _spacing * 0.5 + 2 * _spacing,
            _spacing * 0.5 + 3 * _spacing,
            _spacing * 0.5 + 4 * _spacing,
            _spacing * 0.5 + 5 * _spacing,
            _spacing * 0.5 + 6 * _spacing,
            _spacing * 0.5 + 7 * _spacing,
            _spacing * 0.5 + 8 * _spacing,
            _spacing * 0.5 + 9 * _spacing,
            _spacing * 0.5 + 10 * _spacing,
            _spacing * 0.5 + 11 * _spacing,
            _spacing * 0.5 + 12 * _spacing,
            _spacing * 0.5 + 13 * _spacing,
            _spacing * 0.5 + 14 * _spacing,
            _spacing * 0.5 + 15 * _spacing,
            _spacing * 0.5 + 16 * _spacing,
            _spacing * 0.5 + 17 * _spacing,
            _spacing * 0.5 + 18 * _spacing,
            _spacing * 0.5 + 19 * _spacing,
            _spacing * 0.5 + 20 * _spacing,
            _spacing * 0.5 + 21 * _spacing,
            _spacing * 0.5 + 22 * _spacing,
            _spacing * 0.5 + 23 * _spacing,
            _spacing * 0.5 + 24 * _spacing,
            _spacing * 0.5 + 25 * _spacing,
            _spacing * 0.5 + 26 * _spacing,
            _spacing * 0.5 + 27 * _spacing,
            _spacing * 0.5 + 28 * _spacing,
            _spacing * 0.5 + 29 * _spacing,
            _spacing * 0.5 + 30 * _spacing,
            _spacing * 0.5 + 31 * _spacing,
            _spacing * 0.5 + 32 * _spacing,
            _spacing * 0.5 + 33 * _spacing,
            _spacing * 0.5 + 34 * _spacing,
            _spacing * 0.5 + 35 * _spacing,
            _spacing * 0.5 + 36 * _spacing,
            _spacing * 0.5 + 37 * _spacing,
            _spacing * 0.5 + 38 * _spacing,
            _spacing * 0.5 + 39 * _spacing,
            _spacing * 0.5 + 40 * _spacing,
            _spacing * 0.5 + 41 * _spacing,
            _spacing * 0.5 + 42 * _spacing,
            _spacing * 0.5 + 43 * _spacing,
            _spacing * 0.5 + 44 * _spacing,
            _spacing * 0.5 + 45 * _spacing,
            _spacing * 0.5 + 46 * _spacing,
            _spacing * 0.5 + 47 * _spacing,
            _spacing * 0.5 + 48 * _spacing,
            _spacing * 0.5 + 49 * _spacing,
            _spacing * 0.5 + 50 * _spacing,
            _spacing * 0.5 + 51 * _spacing,
            _spacing * 0.5 + 52 * _spacing,
            _spacing * 0.5 + 53 * _spacing,
            _spacing * 0.5 + 54 * _spacing,
            _spacing * 0.5 + 55 * _spacing,
            _spacing * 0.5 + 56 * _spacing,
            _spacing * 0.5 + 57 * _spacing,
            _spacing * 0.5 + 58 * _spacing,
            _spacing * 0.5 + 59 * _spacing,
            _spacing * 0.5 + 60 * _spacing,
            _spacing * 0.5 + 61 * _spacing,
            _spacing * 0.5 + 62 * _spacing,
            _spacing * 0.5 + 63 * _spacing,
            _spacing * 0.5 + 64 * _spacing,
            _spacing * 0.5 + 65 * _spacing,
            _spacing * 0.5 + 66 * _spacing,
            _spacing * 0.5 + 67 * _spacing,
            _spacing * 0.5 + 68 * _spacing,
            _spacing * 0.5 + 69 * _spacing,
            _spacing * 0.5 + 70 * _spacing,
            _spacing * 0.5 + 71 * _spacing,
            _spacing * 0.5 + 72 * _spacing,
            _spacing * 0.5 + 73 * _spacing,
            _spacing * 0.5 + 74 * _spacing,
            _spacing * 0.5 + 75 * _spacing,
            _spacing * 0.5 + 76 * _spacing,
            _spacing * 0.5 + 77 * _spacing,
            _spacing * 0.5 + 78 * _spacing,
            _spacing * 0.5 + 79 * _spacing,
            _spacing * 0.5 + 80 * _spacing,
            _spacing * 0.5 + 81 * _spacing,
            _spacing * 0.5 + 82 * _spacing,
            _spacing * 0.5 + 83 * _spacing,
            _spacing * 0.5 + 84 * _spacing,
            _spacing * 0.5 + 85 * _spacing,
            _spacing * 0.5 + 86 * _spacing,
            _spacing * 0.5 + 87 * _spacing,
            _spacing * 0.5 + 88 * _spacing,
            _spacing * 0.5 + 89 * _spacing,
            _spacing * 0.5 + 90 * _spacing,
            _spacing * 0.5 + 91 * _spacing,
            _spacing * 0.5 + 92 * _spacing,
            _spacing * 0.5 + 93 * _spacing,
            _spacing * 0.5 + 94 * _spacing,
            _spacing * 0.5 + 95 * _spacing,
            _spacing * 0.5 + 96 * _spacing,
            _spacing * 0.5 + 97 * _spacing,
            _spacing * 0.5 + 98 * _spacing,
            _spacing * 0.5 + 99 * _spacing,
            _spacing * 0.5 + 100 * _spacing,
            _spacing * 0.5 + 101 * _spacing,
            _spacing * 0.5 + 102 * _spacing,
            _spacing * 0.5 + 103 * _spacing,
            _spacing * 0.5 + 104 * _spacing,
            _spacing * 0.5 + 105 * _spacing,
            _spacing * 0.5 + 106 * _spacing,
            _spacing * 0.5 + 107 * _spacing,
            _spacing * 0.5 + 108 * _spacing,
            _spacing * 0.5 + 109 * _spacing,
            _spacing * 0.5 + 110 * _spacing,
            _spacing * 0.5 + 111 * _spacing,
            _spacing * 0.5 + 112 * _spacing,
            _spacing * 0.5 + 113 * _spacing,
            _spacing * 0.5 + 114 * _spacing,
            _spacing * 0.5 + 115 * _spacing,
            _spacing * 0.5 + 116 * _spacing,
            _spacing * 0.5 + 117 * _spacing,
            _spacing * 0.5 + 118 * _spacing,
            _spacing * 0.5 + 119 * _spacing,
        ]
    )

    phi = (np.sqrt(5) + 1) / 2  # golden ratio

    _rabi_rotations = np.array(
        [
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi * (phi - 1) / 3 / np.sqrt(3),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
            2 * np.pi / 5 / np.sqrt(phi + 2),
        ]
    )

    _azimuthal_angles = np.array(
        [
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            np.pi,
            np.pi,
            3 * np.pi / 2,
            np.pi,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
            3 * np.pi / 2,
        ]
    )

    _detuning_rotations = np.array(
        [
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 3 / np.sqrt(3),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
            2 * np.pi * phi / 5 / np.sqrt(phi + 2),
        ]
    )

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_platonic_sequence(
        duration=duration, sequence="Icosahedral", pre_post_rotation=True
    )

    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations, [0, _rabi_rotations.shape[0]], [np.pi / 2, np.pi / 2]
    )
    _azimuthal_angles = np.insert(
        _azimuthal_angles, [0, _azimuthal_angles.shape[0]], [0, np.pi]
    )
    _detuning_rotations = np.insert(
        _detuning_rotations, [0, _detuning_rotations.shape[0]], [0, 0]
    )

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)
