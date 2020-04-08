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
========================
Tests for Predefined DDS
========================
"""


import numpy as np
import pytest


from qctrlopencontrols.exceptions.exceptions import ArgumentsValueError
from qctrlopencontrols import new_predefined_dds
from qctrlopencontrols.dynamic_decoupling_sequences import (
    SPIN_ECHO, CARR_PURCELL, CARR_PURCELL_MEIBOOM_GILL,
    WALSH_SINGLE_AXIS, PERIODIC_SINGLE_AXIS,
    UHRIG_SINGLE_AXIS, QUADRATIC, X_CONCATENATED,
    XY_CONCATENATED)


def test_ramsey():

    """Tests Ramsey sequence
    """

    duration = 10.

    sequence = new_predefined_dds(
        scheme='Ramsey',
        duration=duration)

    _offsets = np.array([])
    _rabi_rotations = np.array([])
    _azimuthal_angles = np.array([])
    _detuning_rotations = np.array([])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme='Ramsey',
        duration=duration,
        pre_post_rotation=True)

    _rabi_rotations = np.array([np.pi/2, np.pi/2])
    _azimuthal_angles = np.array([0., np.pi])
    _detuning_rotations = np.array([0., 0.])

    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_spin_echo():

    """
    Test for Spin Echo Sequence
    """

    duration = 10.

    sequence = new_predefined_dds(
        scheme=SPIN_ECHO,
        duration=duration)

    _offsets = np.array([duration/2.])
    _rabi_rotations = np.array([np.pi])
    _azimuthal_angles = np.array([0])
    _detuning_rotations = np.array([0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=SPIN_ECHO,
        duration=duration,
        pre_post_rotation=True)

    _offsets = np.array([0, duration / 2., duration])
    _rabi_rotations = np.array([np.pi/2, np.pi, np.pi/2])
    _azimuthal_angles = np.array([0, 0, 0])
    _detuning_rotations = np.array([0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_curr_purcell():
    """
    Test for Carr-Purcell (CP) sequence
    """

    duration = 10.
    number_of_offsets = 4

    sequence = new_predefined_dds(
        scheme=CARR_PURCELL,
        duration=duration,
        number_of_offsets=number_of_offsets)

    _spacing = duration/number_of_offsets
    _offsets = np.array([_spacing*0.5, _spacing*0.5+_spacing,
                         _spacing*0.5+2*_spacing, _spacing*0.5+3*_spacing])
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([0, 0, 0, 0])
    _detuning_rotations = np.array([0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=CARR_PURCELL,
        duration=duration,
        number_of_offsets=number_of_offsets,
        pre_post_rotation=True)

    _offsets = np.array([0, _spacing * 0.5, _spacing * 0.5 + _spacing,
                         _spacing * 0.5 + 2 * _spacing, _spacing * 0.5 + 3 * _spacing,
                         duration])
    _rabi_rotations = np.array([np.pi/2, np.pi, np.pi, np.pi, np.pi, np.pi/2])
    _azimuthal_angles = np.array([0, 0, 0, 0, 0, np.pi])
    _detuning_rotations = np.array([0, 0, 0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_curr_purcell_meiboom_sequence():   # pylint: disable=invalid-name
    """
    Test for Carr-Purcell-Meiboom-Sequence (CPMG) sequence
    """

    duration = 10.
    number_of_offsets = 4

    sequence = new_predefined_dds(
        scheme=CARR_PURCELL_MEIBOOM_GILL,
        duration=duration,
        number_of_offsets=number_of_offsets)

    _spacing = duration/number_of_offsets
    _offsets = np.array([_spacing*0.5, _spacing*0.5+_spacing,
                         _spacing*0.5+2*_spacing, _spacing*0.5+3*_spacing])
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
    _detuning_rotations = np.array([0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=CARR_PURCELL_MEIBOOM_GILL,
        duration=duration,
        number_of_offsets=number_of_offsets,
        pre_post_rotation=True)

    _offsets = np.array([0, _spacing * 0.5, _spacing * 0.5 + _spacing,
                         _spacing * 0.5 + 2 * _spacing, _spacing * 0.5 + 3 * _spacing, duration])
    _rabi_rotations = np.array([np.pi/2, np.pi, np.pi, np.pi, np.pi, np.pi/2])
    _azimuthal_angles = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi])
    _detuning_rotations = np.array([0, 0, 0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_uhrig_single_axis_sequence():
    """
    Test for Uhrig Single Axis Sequence
    """

    duration = 10.
    number_of_offsets = 4

    sequence = new_predefined_dds(
        scheme=UHRIG_SINGLE_AXIS,
        duration=duration,
        number_of_offsets=number_of_offsets)

    constant = 0.5 / (number_of_offsets+1)
    _delta_positions = [duration*(np.sin(np.pi*(k+1)*constant))**2
                        for k in range(number_of_offsets)]

    _offsets = np.array(_delta_positions)
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
    _detuning_rotations = np.array([0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=UHRIG_SINGLE_AXIS,
        duration=duration,
        number_of_offsets=number_of_offsets,
        pre_post_rotation=True)

    _offsets = np.array(_delta_positions)
    _offsets = np.insert(_offsets,
                         [0, _offsets.shape[0]],    # pylint: disable=unsubscriptable-object
                         [0, duration])

    _rabi_rotations = np.array([np.pi/2, np.pi, np.pi, np.pi, np.pi, np.pi/2])
    _azimuthal_angles = np.array([0., np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi])
    _detuning_rotations = np.array([0., 0, 0, 0, 0, 0.])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_periodic_single_axis_sequence():      # pylint: disable=invalid-name
    """
    Test for Periodic Single Axis Sequence
    """

    duration = 10.
    number_of_offsets = 4

    sequence = new_predefined_dds(
        scheme=PERIODIC_SINGLE_AXIS,
        duration=duration,
        number_of_offsets=number_of_offsets)

    constant = 1 / (number_of_offsets+1)
    # prepare the offsets for delta comb
    _delta_positions = [duration*k * constant for k in range(1, number_of_offsets + 1)]
    _offsets = np.array(_delta_positions)
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([0, 0, 0, 0])
    _detuning_rotations = np.array([0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=PERIODIC_SINGLE_AXIS,
        duration=duration,
        number_of_offsets=number_of_offsets,
        pre_post_rotation=True)

    _offsets = np.array(_delta_positions)
    _offsets = np.insert(_offsets,
                         [0, _offsets.shape[0]],    # pylint: disable=unsubscriptable-object
                         [0, duration])

    _rabi_rotations = np.array([np.pi/2, np.pi, np.pi, np.pi, np.pi, np.pi/2])
    _azimuthal_angles = np.array([0, 0, 0, 0, 0, np.pi])
    _detuning_rotations = np.array([0, 0, 0, 0, 0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_walsh_single_axis_sequence():
    """
    Test for Periodic Single Axis Sequence
    """

    duration = 10.
    paley_order = 20

    sequence = new_predefined_dds(
        scheme=WALSH_SINGLE_AXIS,
        duration=duration,
        paley_order=paley_order)

    hamming_weight = 5
    samples = 2 ** hamming_weight
    relative_offset = np.arange(1. / (2 * samples), 1., 1. / samples)
    binary_string = np.binary_repr(paley_order)
    binary_order = [int(binary_string[i]) for i in range(hamming_weight)]
    walsh_array = np.ones([samples])
    for i in range(hamming_weight):
        walsh_array *= np.sign(np.sin(2 ** (i + 1) * np.pi
                                      * relative_offset)) ** binary_order[hamming_weight - 1 - i]

    walsh_relative_offsets = []
    for i in range(samples - 1):
        if walsh_array[i] != walsh_array[i + 1]:
            walsh_relative_offsets.append((i + 1) * (1. / samples))
    walsh_relative_offsets = np.array(walsh_relative_offsets, dtype=np.float)
    _offsets = duration * walsh_relative_offsets
    _offsets = np.array(_offsets)

    _rabi_rotations = np.pi * np.ones(_offsets.shape)
    _azimuthal_angles = np.zeros(_offsets.shape)
    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=WALSH_SINGLE_AXIS,
        duration=duration,
        paley_order=paley_order,
        pre_post_rotation=True)

    _offsets = np.insert(_offsets,
                         [0, _offsets.shape[0]],    # pylint: disable=unsubscriptable-object
                         [0, duration])
    _rabi_rotations = np.insert(_rabi_rotations, [0, _rabi_rotations.shape[0]],
                                [np.pi/2, np.pi/2])
    _azimuthal_angles = np.zeros(_offsets.shape)
    _azimuthal_angles[-1] = np.pi
    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_quadratic_sequence():
    """
    Test for Quadratic Sequence
    """

    duration = 10.
    number_inner_offsets = 4
    number_outer_offsets = 4

    sequence = new_predefined_dds(
        scheme=QUADRATIC, duration=duration,
        number_inner_offsets=number_inner_offsets,
        number_outer_offsets=number_outer_offsets)

    _offsets = np.zeros((number_outer_offsets+1, number_inner_offsets + 1))

    constant = 0.5 / (number_outer_offsets + 1)
    _delta_positions = [duration * (np.sin(np.pi * (k + 1) * constant)) ** 2
                        for k in range(number_outer_offsets)]

    _outer_offsets = np.array(_delta_positions)
    _offsets[0:number_outer_offsets, -1] = _outer_offsets

    _outer_offsets = np.insert(
        _outer_offsets,
        [0, _outer_offsets.shape[0]],   # pylint: disable=unsubscriptable-object
        [0, duration])
    _inner_durations = _outer_offsets[1:] - _outer_offsets[0:-1]

    constant = 0.5 / (number_inner_offsets+1)
    _delta_positions = [(np.sin(np.pi * (k + 1) * constant)) ** 2
                        for k in range(number_inner_offsets)]
    _delta_positions = np.array(_delta_positions)
    for inner_sequence_idx in range(_inner_durations.shape[0]):
        _inner_deltas = _inner_durations[inner_sequence_idx] * _delta_positions
        _inner_deltas = _outer_offsets[inner_sequence_idx] + _inner_deltas
        _offsets[inner_sequence_idx, 0:number_inner_offsets] = _inner_deltas

    _rabi_rotations = np.zeros(_offsets.shape)
    _detuning_rotations = np.zeros(_offsets.shape)

    _rabi_rotations[0:number_outer_offsets, -1] = np.pi
    _detuning_rotations[0:(number_outer_offsets+1), 0:number_inner_offsets] = np.pi

    _offsets = np.reshape(_offsets, (-1,))
    _rabi_rotations = np.reshape(_rabi_rotations, (-1,))
    _detuning_rotations = np.reshape(_detuning_rotations, (-1,))

    _offsets = _offsets[0:-1]
    _rabi_rotations = _rabi_rotations[0:-1]
    _detuning_rotations = _detuning_rotations[0:-1]

    _azimuthal_angles = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=QUADRATIC, duration=duration,
        number_inner_offsets=number_inner_offsets,
        number_outer_offsets=number_outer_offsets,
        pre_post_rotation=True)

    _offsets = np.insert(_offsets, [0, _offsets.shape[0]], [0, duration])
    _rabi_rotations = np.insert(_rabi_rotations, [0, _rabi_rotations.shape[0]],
                                [np.pi/2, np.pi/2])
    _detuning_rotations = np.insert(_detuning_rotations, [0, _detuning_rotations.shape[0]],
                                    [0, 0])

    _azimuthal_angles = np.zeros(_offsets.shape)
    _azimuthal_angles[-1] = np.pi

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_xconcatenated_sequence():
    """
    Test X-CDD Sequence
    """

    duration = 10.
    concatenation_order = 3

    sequence = new_predefined_dds(
        scheme=X_CONCATENATED,
        duration=duration,
        concatenation_order=concatenation_order)

    _spacing = duration/(2**concatenation_order)
    _offsets = [_spacing, 3*_spacing, 4 * _spacing, 5 * _spacing, 7 * _spacing]
    _offsets = np.array(_offsets)
    _rabi_rotations = np.pi * np.ones(_offsets.shape)

    _azimuthal_angles = np.zeros(_offsets.shape)
    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=X_CONCATENATED,
        duration=duration,
        concatenation_order=concatenation_order,
        pre_post_rotation=True)

    _offsets = np.insert(
        _offsets,
        [0, _offsets.shape[0]], # pylint: disable=unsubscriptable-object
        [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations,
        [0, _rabi_rotations.shape[0]],  # pylint: disable=unsubscriptable-object
        [np.pi/2, np.pi/2])
    _azimuthal_angles = np.zeros(_offsets.shape)
    _detuning_rotations = np.zeros(_offsets.shape)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_xyconcatenated_sequence():
    """
    Test XY4-CDD Sequence
    """

    duration = 10.
    concatenation_order = 2

    sequence = new_predefined_dds(
        scheme=XY_CONCATENATED,
        duration=duration,
        concatenation_order=concatenation_order)

    _spacing = duration / (2 ** (concatenation_order*2))
    _offsets = [_spacing, 2*_spacing, 3 * _spacing, 4 * _spacing,
                5 * _spacing, 6 * _spacing, 7 * _spacing, 9 * _spacing,
                10 * _spacing, 11 * _spacing, 12 * _spacing, 13 * _spacing,
                14 * _spacing, 15 * _spacing]
    _offsets = np.array(_offsets)
    _rabi_rotations = [np.pi, np.pi, np.pi, 0., np.pi, np.pi, np.pi,
                       np.pi, np.pi, np.pi, 0, np.pi, np.pi, np.pi]
    _rabi_rotations = np.array(_rabi_rotations)
    _azimuthal_angles = [0, np.pi/2, 0, 0, 0, np.pi/2, 0, 0, np.pi/2, 0, 0, 0, np.pi/2, 0]
    _azimuthal_angles = np.array(_azimuthal_angles)
    _detuning_rotations = [0, 0, 0, np.pi, 0, 0, 0, 0, 0, 0, np.pi, 0, 0, 0]
    _detuning_rotations = np.array(_detuning_rotations)

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)

    sequence = new_predefined_dds(
        scheme=XY_CONCATENATED,
        duration=duration,
        concatenation_order=concatenation_order,
        pre_post_rotation=True)

    _offsets = np.insert(_offsets,
                         [0, _offsets.shape[0]],    # pylint: disable=unsubscriptable-object
                         [0, duration])
    _rabi_rotations = np.insert(
        _rabi_rotations,
        [0, _rabi_rotations.shape[0]],  # pylint: disable=unsubscriptable-object
        [np.pi/2, np.pi/2])
    _azimuthal_angles = np.insert(
        _azimuthal_angles,
        [0, _azimuthal_angles.shape[0]],    # pylint: disable=unsubscriptable-object
        [0, np.pi])
    _detuning_rotations = np.insert(
        _detuning_rotations,
        [0, _detuning_rotations.shape[0]],  # pylint: disable=unsubscriptable-object
        [0, 0])

    assert np.allclose(_offsets, sequence.offsets)
    assert np.allclose(_rabi_rotations, sequence.rabi_rotations)
    assert np.allclose(_azimuthal_angles, sequence.azimuthal_angles)
    assert np.allclose(_detuning_rotations, sequence.detuning_rotations)


def test_attribute_values():
    """
    Test for the correctness of the attribute values
    """

    # Check that errors are raised correctly

    # duration cannot be <= 0
    with pytest.raises(ArgumentsValueError):
        _ = new_predefined_dds(scheme=SPIN_ECHO, duration=-2)

        # number_of_offsets cannot be <= 0
        _ = new_predefined_dds(
            scheme=CARR_PURCELL_MEIBOOM_GILL, duration=2,
            number_of_offsets=-1)
        # for QDD, none of the offsets can be <=0
        _ = new_predefined_dds(
            scheme=QUADRATIC, duration=2,
            number_inner_offsets=-1, number_outer_offsets=2)
        _ = new_predefined_dds(
            scheme=QUADRATIC, duration=2,
            number_inner_offsets=1, number_outer_offsets=-2)
        _ = new_predefined_dds(
            scheme=QUADRATIC, duration=2,
            number_inner_offsets=-1, number_outer_offsets=-2)

        # for x-cdd and xy-cdd concatenation_order cannot be <=0
        _ = new_predefined_dds(
            scheme=X_CONCATENATED, duration=2,
            concatenation_order=-1)
        _ = new_predefined_dds(
            scheme=X_CONCATENATED, duration=-2,
            concatenation_order=1)
        _ = new_predefined_dds(
            scheme=X_CONCATENATED, duration=-2,
            concatenation_order=-1)
        _ = new_predefined_dds(
            scheme=XY_CONCATENATED, duration=2,
            concatenation_order=-1)
        _ = new_predefined_dds(
            scheme=XY_CONCATENATED, duration=-2,
            concatenation_order=1)
        _ = new_predefined_dds(
            scheme=XY_CONCATENATED, duration=-2,
            concatenation_order=-1)

def _pulses_produce_identity(sequence):
    """
    Tests if the pulses of a DDS sequence produce an identity in absence of noise.
    We check this by creating the unitary of each pulse and then multiplying them
    by each other to check the complete evolution.
    """
    sigma_x = np.array([[0., 1.], [1., 0.]])
    sigma_y = np.array([[0., -1.j], [1.j, 0.]])
    sigma_z = np.array([[1., 0.], [0., -1.]])

    # The unitary evolution due to an instantaneous pulse can be written as
    # U = cos(|n|) I -i sin(|n|) *(n_x sigma_x + n_y sigma_y + n_z sigma_z)/|n|
    # where n is a vector with components
    # n_x = rabi * cos(azimuthal)/2
    # n_y = rabi * sin(azimuthal)/2
    # n_z = detuning/2

    matrix_product = np.identity(2)
    for rabi, azimuth, detuning in zip(sequence.rabi_rotations,
                                       sequence.azimuthal_angles,
                                       sequence.detuning_rotations):
        n_x = rabi * np.cos(azimuth)/2.
        n_y = rabi * np.sin(azimuth)/2.
        n_z = detuning/2.
        mod_n = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        unitary = (
            np.cos(mod_n) * np.identity(2)
            -1.j * (np.sin(mod_n)*n_x/mod_n) * sigma_x
            -1.j * (np.sin(mod_n)*n_y/mod_n) * sigma_y
            -1.j * (np.sin(mod_n)*n_z/mod_n) * sigma_z
        )
        matrix_product = np.matmul(unitary, matrix_product)

    # Remove global phase
    matrix_product *= np.exp(-1.j* np.angle(matrix_product[0][0]))

    expected_matrix_product = np.identity(2)

    return np.allclose(matrix_product, expected_matrix_product)

def test_if_ramsey_sequence_is_identity():
    """
    Tests if the product of the pulses in the Ramsey sequence with pre/post
    pi/2-pulses is an identity.
    """
    ramsey_sequence = new_predefined_dds(
        scheme='Ramsey',
        duration=10.,
        pre_post_rotation=True)

    assert _pulses_produce_identity(ramsey_sequence)

def test_if_spin_echo_sequence_is_identity():
    """
    Tests if the product of the pulses in a Spin Echo sequence with pre/post
    pi/2-pulses is an identity.
    """
    spin_echo_sequence = new_predefined_dds(
        scheme=SPIN_ECHO,
        duration=10.,
        pre_post_rotation=True)

    assert _pulses_produce_identity(spin_echo_sequence)

def test_if_carr_purcell_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a Carr-Purcell sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is odd.
    """
    odd_carr_purcell_sequence = new_predefined_dds(
        scheme=CARR_PURCELL,
        duration=10.,
        number_of_offsets=7,
        pre_post_rotation=True)

    assert _pulses_produce_identity(odd_carr_purcell_sequence)

def test_if_carr_purcell_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a Carr-Purcell sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_carr_purcell_sequence = new_predefined_dds(
        scheme=CARR_PURCELL,
        duration=10.,
        number_of_offsets=8,
        pre_post_rotation=True)

    assert _pulses_produce_identity(even_carr_purcell_sequence)

def test_if_cpmg_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a CPMG sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is odd.
    """
    odd_cpmg_sequence = new_predefined_dds(
        scheme=CARR_PURCELL_MEIBOOM_GILL,
        duration=10.,
        number_of_offsets=7,
        pre_post_rotation=True)

    assert _pulses_produce_identity(odd_cpmg_sequence)

def test_if_cpmg_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a CPMG sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_cpmg_sequence = new_predefined_dds(
        scheme=CARR_PURCELL_MEIBOOM_GILL,
        duration=10.,
        number_of_offsets=8,
        pre_post_rotation=True)

    assert _pulses_produce_identity(even_cpmg_sequence)

def test_if_uhrig_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in an Uhrig sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is odd.
    """
    odd_uhrig_sequence = new_predefined_dds(
        scheme=UHRIG_SINGLE_AXIS,
        duration=10.,
        number_of_offsets=7,
        pre_post_rotation=True)

    assert _pulses_produce_identity(odd_uhrig_sequence)

def test_if_uhrig_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in an Uhrig sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_uhrig_sequence = new_predefined_dds(
        scheme=UHRIG_SINGLE_AXIS,
        duration=10.,
        number_of_offsets=8,
        pre_post_rotation=True)

    assert _pulses_produce_identity(even_uhrig_sequence)

def test_if_periodic_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a periodic DDS with pre/post
    pi/2-pulses is an identity, when the number of pulses is odd.
    """
    odd_periodic_sequence = new_predefined_dds(
        scheme=PERIODIC_SINGLE_AXIS,
        duration=10.,
        number_of_offsets=7,
        pre_post_rotation=True)

    assert _pulses_produce_identity(odd_periodic_sequence)

def test_if_periodic_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a periodic DDS with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_periodic_sequence = new_predefined_dds(
        scheme=PERIODIC_SINGLE_AXIS,
        duration=10.,
        number_of_offsets=8,
        pre_post_rotation=True)

    assert _pulses_produce_identity(even_periodic_sequence)

def test_if_walsh_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a Walsh sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is odd.
    """
    odd_walsh_sequence = new_predefined_dds(
        scheme=WALSH_SINGLE_AXIS,
        duration=10.,
        paley_order=7,
        pre_post_rotation=True)

    # A Walsh sequence with paley_order 7 has 5 pi-pulses + 2 pi/2-pulses,
    # see https://arxiv.org/pdf/1109.6002.pdf
    assert len(odd_walsh_sequence.offsets) == 5 + 2

    assert _pulses_produce_identity(odd_walsh_sequence)

def test_if_walsh_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the number of pulses is even.
    """
    even_walsh_sequence = new_predefined_dds(
        scheme=WALSH_SINGLE_AXIS,
        duration=10.,
        paley_order=6,
        pre_post_rotation=True)

    # A Walsh sequence with paley_order 6 has 4 pi-pulses + 2 pi/2-pulses,
    # see https://arxiv.org/pdf/1109.6002.pdf
    assert len(even_walsh_sequence.offsets) == 4 + 2

    assert _pulses_produce_identity(even_walsh_sequence)

def test_if_quadratic_sequence_with_odd_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the total number of pulses is odd.
    """
    odd_quadratic_sequence = new_predefined_dds(
        scheme=QUADRATIC,
        duration=10.,
        number_inner_offsets=7,
        number_outer_offsets=7,
        pre_post_rotation=True)

    # n_outer + n_inner*(n_outer+1) pi-pulses + 2 pi/2-pulses
    # total number here is odd
    assert len(odd_quadratic_sequence.offsets) == 7 + 7 * (7+1) + 2

    assert _pulses_produce_identity(odd_quadratic_sequence)


def test_if_quadratic_sequence_with_even_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the total number of pulses is even.
    """
    even_quadratic_sequence = new_predefined_dds(
        scheme=QUADRATIC,
        duration=10.,
        number_inner_offsets=8,
        number_outer_offsets=8,
        pre_post_rotation=True)

    # n_outer + n_inner*(n_outer+1) pi-pulses + 2 pi/2-pulses
    # total number here is even
    assert len(even_quadratic_sequence.offsets) == 8 + 8 * (8+1) + 2

    assert _pulses_produce_identity(even_quadratic_sequence)

def test_if_quadratic_sequence_with_odd_inner_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the total number of inner pulses is odd.
    """
    inner_odd_quadratic_sequence = new_predefined_dds(
        scheme=QUADRATIC,
        duration=10.,
        number_inner_offsets=7,
        number_outer_offsets=8,
        pre_post_rotation=True)

    # n_outer + n_inner*(n_outer+1) pi-pulses + 2 pi/2-pulses
    # total number here is odd
    assert len(inner_odd_quadratic_sequence.offsets) == 8 + 7 * (8+1) + 2

    assert _pulses_produce_identity(inner_odd_quadratic_sequence)


def test_if_quadratic_sequence_with_even_inner_pulses_is_identity():
    """
    Tests if the product of the pulses in a quadratic sequence with pre/post
    pi/2-pulses is an identity, when the total number of inner pulses is even.
    """
    inner_even_quadratic_sequence = new_predefined_dds(
        scheme=QUADRATIC,
        duration=10.,
        number_inner_offsets=8,
        number_outer_offsets=7,
        pre_post_rotation=True)

    # n_outer + n_inner*(n_outer+1) pi-pulses + 2 pi/2-pulses
    # total number here is even
    assert len(inner_even_quadratic_sequence.offsets) == 7 + 8 * (7+1) + 2

    assert _pulses_produce_identity(inner_even_quadratic_sequence)

def test_if_x_concatenated_sequence_is_identity():
    """
    Tests if the product of the pulses in an X concatenated sequence with pre/post
    pi/2-pulses is an identity.
    """
    x_concat_sequence = new_predefined_dds(
        scheme=X_CONCATENATED,
        duration=10.,
        concatenation_order=4,
        pre_post_rotation=True)

    assert _pulses_produce_identity(x_concat_sequence)

def test_if_xy_concatenated_sequence_is_identity():
    """
    Tests if the product of the pulses in an XY concatenated sequence with pre/post
    pi/2-pulses is an identity.
    """
    xy_concat_sequence = new_predefined_dds(
        scheme=XY_CONCATENATED,
        duration=10.,
        concatenation_order=4,
        pre_post_rotation=True)

    assert _pulses_produce_identity(xy_concat_sequence)
