# Copyright 2020 Q-CTRL Pty Ltd & Q-CTRL Inc. All rights reserved.
#
# Licensed under the Q-CTRL Terms of service (the "License"). Unauthorized
# copying or use of this file, via any medium, is strictly prohibited.
# Proprietary and confidential. You may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://q-ctrl.com/terms
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS. See the
# License for the specific language.

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
    _azimuthal_angles = np.array([0., 0.])
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
    _azimuthal_angles = np.array([0, 0, 0, 0, 0, 0])
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
    _azimuthal_angles = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0])
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
    _azimuthal_angles = np.array([0., np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.])
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
    _azimuthal_angles = np.array([0, 0, 0, 0, 0, 0])
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
        [0, 0])
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
