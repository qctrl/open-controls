# Copyright 2022 Q-CTRL
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
Tests for Dynamical Decoupling Sequences.
"""


import os

import numpy as np
import pytest

from qctrlopencontrols import (
    DynamicDecouplingSequence,
    convert_dds_to_driven_control,
)
from qctrlopencontrols.exceptions import ArgumentsValueError


def _remove_file(filename):
    """
    Removes the file after test done.
    """

    if os.path.exists(filename):
        os.remove(filename)
    else:
        raise IOError(f"Could not find file {filename}")


def test_dynamical_decoupling_sequence():
    """
    Tests the Dynamic Decoupling Sequence class.
    """

    _duration = 2.0
    _offsets = np.array([0.25, 0.5, 0.75, 1.00, 1.25, 1.50, 1.75])
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
    _azimthal_angles = 0.5 * np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
    _detuning_rotations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _name = "test_sequence"

    sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    assert sequence.duration == _duration
    assert np.allclose(sequence.offsets, _offsets)
    assert np.allclose(sequence.rabi_rotations, _rabi_rotations)
    assert np.allclose(sequence.azimuthal_angles, _azimthal_angles)
    assert np.allclose(sequence.detuning_rotations, _detuning_rotations)
    assert sequence.name == _name

    _repr_string = f"{sequence.__class__.__name__!s}("

    attributes = {
        "duration": sequence.duration,
        "offsets": sequence.offsets,
        "rabi_rotations": sequence.rabi_rotations,
        "azimuthal_angles": sequence.azimuthal_angles,
        "detuning_rotations": sequence.detuning_rotations,
        "name": sequence.name,
    }

    attributes_string = ",".join(
        f"{attribute}={repr(getattr(sequence, attribute))}" for attribute in attributes
    )
    _repr_string += attributes_string
    _repr_string += ")"

    assert repr(sequence) == _repr_string

    _duration = 2.0
    _offsets = np.array([0.0, 0.25, 0.5, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    _rabi_rotations = np.array(
        [np.pi / 2, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi / 2]
    )
    _azimthal_angles = 0.5 * np.array(
        [0.0, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 0.0]
    )
    _detuning_rotations = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _name = "test_sequence"

    sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    assert sequence.duration == _duration
    assert np.allclose(sequence.offsets, _offsets)
    assert np.allclose(sequence.rabi_rotations, _rabi_rotations)
    assert np.allclose(sequence.azimuthal_angles, _azimthal_angles)
    assert np.allclose(sequence.detuning_rotations, _detuning_rotations)
    assert sequence.name == _name

    with pytest.raises(ArgumentsValueError):

        # duration cannot be negative
        _ = DynamicDecouplingSequence(
            duration=-2.0,
            offsets=2.0 / 2000 * np.ones((2000, 1)),
            rabi_rotations=np.pi * np.ones((2000, 1)),
            azimuthal_angles=np.ones((2000, 1)),
            detuning_rotations=np.zeros((2000, 1)),
        )

    with pytest.raises(ArgumentsValueError):

        # rabi rotations cannot be negative
        _ = DynamicDecouplingSequence(
            duration=2.0,
            offsets=np.ones((2, 1)),
            rabi_rotations=np.asarray([1, -1]),
            azimuthal_angles=np.ones((2, 1)),
            detuning_rotations=np.zeros((2, 1)),
        )


def test_sequence_plot():
    """
    Tests the plot data of sequences.
    """

    # An arbitrary sequence - may not conform to any of the predefined ones
    _offsets = np.array([0, 0.25, 0.5, 0.75, 1.00])
    _rabi_rotations = np.array([0, np.pi, 0, np.pi, 0])
    _azimuthal_angle = np.array([0, 0, 0, 0, 0])
    _detuning_rotations = np.array([0, 0, 0, 0, 0])

    seq = DynamicDecouplingSequence(
        duration=1.0,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angle,
        detuning_rotations=_detuning_rotations,
    )

    plot_data = seq.export()

    _plot_rabi_offsets = [pulse["offset"] for pulse in plot_data["Rabi"]]
    _plot_detuning_offsets = [pulse["offset"] for pulse in plot_data["Detuning"]]
    _plot_rabi_rotations = [pulse["rotation"] for pulse in plot_data["Rabi"]]
    _plot_detuning_rotations = [pulse["rotation"] for pulse in plot_data["Detuning"]]

    assert np.allclose(_plot_rabi_offsets, _offsets)
    assert np.allclose(_plot_detuning_offsets, _offsets)

    assert np.allclose(np.abs(_plot_rabi_rotations), _rabi_rotations)
    assert np.allclose(np.angle(_plot_rabi_rotations), _azimuthal_angle)

    assert np.allclose(_plot_detuning_rotations, _detuning_rotations)

    # with both X and Y pi
    _offsets = np.array([0, 0.25, 0.5, 0.75, 1.00])
    _rabi_rotations = np.array([0, np.pi, 0, np.pi, 0])
    _azimuthal_angle = np.array([0, np.pi / 2, 0, np.pi / 2, 0])
    _detuning_rotations = np.array([0, 0, 0, 0, 0])

    seq = DynamicDecouplingSequence(
        duration=1.0,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angle,
        detuning_rotations=_detuning_rotations,
    )

    plot_data = seq.export()

    _plot_rabi_offsets = [pulse["offset"] for pulse in plot_data["Rabi"]]
    _plot_detuning_offsets = [pulse["offset"] for pulse in plot_data["Detuning"]]
    _plot_rabi_rotations = [pulse["rotation"] for pulse in plot_data["Rabi"]]
    _plot_detuning_rotations = [pulse["rotation"] for pulse in plot_data["Detuning"]]

    assert np.allclose(_plot_rabi_offsets, _offsets)
    assert np.allclose(_plot_detuning_offsets, _offsets)

    assert np.allclose(np.abs(_plot_rabi_rotations), _rabi_rotations)
    assert np.allclose(np.angle(_plot_rabi_rotations), _azimuthal_angle)

    assert np.allclose(_plot_detuning_rotations, _detuning_rotations)


def test_pretty_string_format():

    """
    Tests `__str__` of the dynamic decoupling sequence.
    """

    _duration = 1.0
    _offsets = np.array([0.0, 0.5, 1.0])
    _rabi_rotations = np.array([0.0, np.pi, 0.0])
    _azimuthal_angles = 0.5 * np.array([0.0, np.pi / 2, np.pi])
    _detuning_rotations = np.array([np.pi / 4, 0.0, 0.0])
    _name = "test_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _pretty_string = ["test_sequence:"]
    _pretty_string.append(f"Duration = {_duration}")
    _pretty_string.append(
        f"Offsets = [{_offsets[0]}, {_offsets[1]}, {_offsets[2]}] × {_duration}"
    )
    _pretty_string.append(
        f"Rabi Rotations = [{_rabi_rotations[0] / np.pi},"
        f" {_rabi_rotations[1] / np.pi}, {_rabi_rotations[2]/ np.pi}] × pi"
    )
    _pretty_string.append(
        f"Azimuthal Angles = [{_azimuthal_angles[0] / np.pi},"
        f" {_azimuthal_angles[1] / np.pi}, {_azimuthal_angles[2] / np.pi}] × pi"
    )
    _pretty_string.append(
        f"Detuning Rotations = [{_detuning_rotations[0] / np.pi},"
        f" {_detuning_rotations[1] / np.pi}, {_detuning_rotations[2] / np.pi}] × pi"
    )

    expected_string = "\n".join(_pretty_string)

    assert expected_string == str(dd_sequence)

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
    )

    _pretty_string = []
    _pretty_string.append(f"Duration = {_duration}")
    _pretty_string.append(
        f"Offsets = [{_offsets[0]}, {_offsets[1]}, {_offsets[2]}] × {_duration}"
    )
    _pretty_string.append(
        f"Rabi Rotations = [{_rabi_rotations[0] / np.pi},"
        f" {_rabi_rotations[1] / np.pi}, {_rabi_rotations[2]/ np.pi}] × pi"
    )
    _pretty_string.append(
        f"Azimuthal Angles = [{_azimuthal_angles[0] / np.pi},"
        f" {_azimuthal_angles[1] / np.pi}, {_azimuthal_angles[2]/ np.pi}] × pi"
    )
    _pretty_string.append(
        f"Detuning Rotations = [{_detuning_rotations[0] / np.pi},"
        f" {_detuning_rotations[1] / np.pi}, {_detuning_rotations[2] / np.pi}] × pi"
    )
    expected_string = "\n".join(_pretty_string)

    assert expected_string == str(dd_sequence)


def test_conversion_to_driven_controls():

    """
    Tests the method to convert a DDS to Driven Control.
    """
    _duration = 2.0
    _offsets = 2 * np.array([0.25, 0.5, 0.75])
    _rabi_rotations = np.array([np.pi, 0.0, np.pi])
    _azimuthal_angles = np.array([np.pi / 2, 0.0, 0.0])
    _detuning_rotations = np.array([0.0, np.pi, 0.0])
    _name = "test_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name,
    )

    assert np.allclose(
        driven_control.rabi_rates,
        np.array([0.0, _maximum_rabi_rate, 0.0, 0.0, 0.0, _maximum_rabi_rate, 0.0]),
    )
    assert np.allclose(
        driven_control.azimuthal_angles,
        np.array([0.0, _azimuthal_angles[0], 0.0, 0.0, 0.0, _azimuthal_angles[2], 0.0]),
    )
    assert np.allclose(
        driven_control.detunings,
        np.array([0.0, 0.0, 0.0, _maximum_detuning_rate, 0.0, 0.0, 0]),
    )
    assert np.allclose(
        driven_control.durations,
        np.array([4.75e-1, 5e-2, 4.5e-1, 5e-2, 4.5e-1, 5e-2, 4.75e-1]),
    )


def test_conversion_of_pi_2_pulses_to_driven_controls():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    pi/2-pulses in the x, y, and z directions.
    """
    _duration = 6.0
    _offsets = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    _rabi_rotations = np.array([np.pi / 2, 0.0, np.pi / 2, np.pi / 2, 0.0, np.pi / 2])
    _azimuthal_angles = np.array([np.pi / 2, 0.0, 0.0, -np.pi / 2, 0, np.pi])
    _detuning_rotations = np.array([0.0, np.pi / 2, 0.0, 0.0, -np.pi / 2, 0])
    _name = "pi2_pulse_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name,
    )

    assert np.allclose(
        driven_control.rabi_rates,
        np.array(
            [
                0.0,
                _maximum_rabi_rate,
                0.0,
                0.0,
                0.0,
                _maximum_rabi_rate,
                0.0,
                _maximum_rabi_rate,
                0.0,
                0.0,
                0.0,
                _maximum_rabi_rate,
                0.0,
            ]
        ),
    )
    assert np.allclose(
        driven_control.azimuthal_angles,
        np.array(
            [
                0.0,
                _azimuthal_angles[0],
                0.0,
                0.0,
                0.0,
                _azimuthal_angles[2],
                0.0,
                _azimuthal_angles[3],
                0.0,
                0.0,
                0.0,
                _azimuthal_angles[5],
                0.0,
            ]
        ),
    )
    assert np.allclose(
        driven_control.detunings,
        np.array(
            [
                0.0,
                0.0,
                0.0,
                _maximum_detuning_rate,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -_maximum_detuning_rate,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )
    assert np.allclose(
        driven_control.durations,
        np.array(
            [
                4.875e-1,
                2.5e-2,
                9.75e-1,
                2.5e-2,
                9.75e-1,
                2.5e-2,
                9.75e-1,
                2.5e-2,
                9.75e-1,
                2.5e-2,
                9.75e-1,
                2.5e-2,
                4.875e-1,
            ]
        ),
    )


def test_conversion_of_x_pi_2_pulses_at_extremities():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    X pi/2-pulses in inverse directions, in the beginning and end of the sequence.
    """
    _duration = 1.0
    _offsets = np.array([0.0, _duration])
    _rabi_rotations = np.array([np.pi / 2, np.pi / 2])
    _azimuthal_angles = np.array([np.pi, 0])
    _detuning_rotations = np.array([0.0, 0])
    _name = "x_pi2_pulse_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name,
    )

    assert np.allclose(
        driven_control.rabi_rates,
        np.array([_maximum_rabi_rate, 0.0, _maximum_rabi_rate]),
    )
    assert np.allclose(
        driven_control.azimuthal_angles,
        np.array([_azimuthal_angles[0], 0.0, _azimuthal_angles[1]]),
    )
    assert np.allclose(driven_control.detunings, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(
        driven_control.durations, np.array([2.5e-2, _duration - 2 * 2.5e-2, 2.5e-2])
    )


def test_conversion_of_y_pi_2_pulses_at_extremities():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    Y pi/2-pulses in inverse directions, in the beginning and end of the sequence.
    """
    _duration = 1.0
    _offsets = np.array([0.0, _duration])
    _rabi_rotations = np.array([np.pi / 2, np.pi / 2])
    _azimuthal_angles = np.array([-np.pi / 2, np.pi / 2])
    _detuning_rotations = np.array([0.0, 0])
    _name = "y_pi2_pulse_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name,
    )

    assert np.allclose(
        driven_control.rabi_rates,
        np.array([_maximum_rabi_rate, 0.0, _maximum_rabi_rate]),
    )
    assert np.allclose(
        driven_control.azimuthal_angles,
        np.array([_azimuthal_angles[0], 0.0, _azimuthal_angles[1]]),
    )
    assert np.allclose(driven_control.detunings, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(
        driven_control.durations, np.array([2.5e-2, _duration - 2 * 2.5e-2, 2.5e-2])
    )


def test_conversion_of_z_pi_2_pulses_at_extremities():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    Z pi/2-pulses in inverse directions, in the beginning and end of the sequence.
    """
    _duration = 1.0
    _offsets = np.array([0.0, _duration])
    _rabi_rotations = np.array([0, 0])
    _azimuthal_angles = np.array([0, 0])
    _detuning_rotations = np.array([-np.pi / 2, np.pi / 2])
    _name = "z_pi2_pulse_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name,
    )

    assert np.allclose(driven_control.rabi_rates, np.array([0, 0.0, 0]))
    assert np.allclose(driven_control.azimuthal_angles, np.array([0, 0.0, 0]))
    assert np.allclose(
        driven_control.detunings,
        np.array([-_maximum_detuning_rate, 0.0, _maximum_detuning_rate]),
    )
    assert np.allclose(
        driven_control.durations, np.array([2.5e-2, _duration - 2 * 2.5e-2, 2.5e-2])
    )


def test_conversion_of_pulses_with_arbitrary_rabi_rotations():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    Y pulses with rabi rotations that assume arbitrary values between 0 and pi.
    """
    _duration = 3.0
    _offsets = np.array([0.5, 1.5, 2.5])
    _rabi_rotations = np.array([1.0, 2.0, 3.0])
    _azimuthal_angles = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
    _detuning_rotations = np.array([0.0, 0.0, 0.0])
    _name = "arbitrary_rabi_rotation_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20
    _maximum_detuning_rate = 10
    minimum_segment_duration = 0.1

    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        minimum_segment_duration=minimum_segment_duration,
        name=_name,
    )

    expected_rabi_rates = [0.0, 10.0, 0.0, 20.0, 0.0, 20.0, 0.0]
    expected_azimuthal_angles = [0.0, np.pi / 2, 0.0, np.pi / 2, 0.0, np.pi / 2, 0.0]
    expected_detuning_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected_durations = [
        0.5 - 0.1 / 2,
        0.1,
        1.0 - 0.1 / 2 - 0.1 / 2,
        0.1,
        1.0 - 0.1 / 2 - 0.15 / 2,
        0.15,
        0.5 - 0.15 / 2,
    ]

    assert np.allclose(driven_control.rabi_rates, expected_rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, expected_azimuthal_angles)
    assert np.allclose(driven_control.detunings, expected_detuning_rates)
    assert np.allclose(driven_control.durations, expected_durations)

    # check explicitly that minimum segment duration is respected
    assert _all_greater_or_close(driven_control.duration, minimum_segment_duration)


def test_conversion_of_pulses_with_arbitrary_azimuthal_angles():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    pi-pulses with azimuthal angles that assume arbitrary values between 0 and pi/2.
    """
    _duration = 3.0
    _offsets = np.array([0.5, 1.5, 2.5])
    _rabi_rotations = np.array([np.pi, np.pi, np.pi])
    _azimuthal_angles = np.array([0.5, 1.0, 1.5])
    _detuning_rotations = np.array([0.0, 0.0, 0.0])
    _name = "arbitrary_azimuthal_angle_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 10 * np.pi
    minimum_segment_duration = 0.1

    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        minimum_segment_duration=minimum_segment_duration,
        name=_name,
    )

    expected_rabi_rates = [0.0, 10.0 * np.pi, 0.0, 10.0 * np.pi, 0.0, 10.0 * np.pi, 0.0]
    expected_azimuthal_angles = [0.0, 0.5, 0.0, 1.0, 0.0, 1.5, 0.0]
    expected_detuning_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected_durations = [
        0.5 - 0.1 / 2,
        0.1,
        1.0 - 0.1 / 2 - 0.1 / 2,
        0.1,
        1.0 - 0.1 / 2 - 0.1 / 2,
        0.1,
        0.5 - 0.1 / 2,
    ]

    assert np.allclose(driven_control.rabi_rates, expected_rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, expected_azimuthal_angles)
    assert np.allclose(driven_control.detunings, expected_detuning_rates)
    assert np.allclose(driven_control.durations, expected_durations)

    # check explicitly that minimum segment duration is respected
    assert _all_greater_or_close(driven_control.duration, minimum_segment_duration)


def test_conversion_of_pulses_with_arbitrary_detuning_rotations():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    Z pulses with detuning rotations that assume arbitrary values between 0 and pi.
    """
    _duration = 3.0
    _offsets = np.array([0.5, 1.5, 2.5])
    _rabi_rotations = np.array([0.0, 0.0, 0.0])
    _azimuthal_angles = np.array([0.0, 0.0, 0.0])
    _detuning_rotations = np.array([1.0, 2.0, 3.0])
    _name = "arbitrary_detuning_rotation_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20
    _maximum_detuning_rate = 10
    minimum_segment_duration = 0.2

    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        minimum_segment_duration=minimum_segment_duration,
        name=_name,
    )

    expected_rabi_rates = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected_azimuthal_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected_detuning_rates = [0.0, 5.0, 0.0, 10.0, 0.0, 10.0, 0.0]
    expected_durations = [
        0.5 - 0.2 / 2,
        0.2,
        1.0 - 0.2 / 2 - 0.2 / 2,
        0.2,
        1.0 - 0.2 / 2 - 0.3 / 2,
        0.3,
        0.5 - 0.3 / 2,
    ]

    assert np.allclose(driven_control.rabi_rates, expected_rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, expected_azimuthal_angles)
    assert np.allclose(driven_control.detunings, expected_detuning_rates)
    assert np.allclose(driven_control.durations, expected_durations)

    # check explicitly that minimum segment duration is respected
    assert _all_greater_or_close(driven_control.duration, minimum_segment_duration)


def test_conversion_of_tightly_packed_sequence():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    a sequence tightly packed with pulses, where there is no time for a gap between
    the pi/2-pulses and the adjacent pi-pulses.
    """
    # create a sequence containing 2 pi-pulses and 2 pi/2-pulses at the extremities
    dynamic_decoupling_sequence = DynamicDecouplingSequence(
        duration=0.2,
        offsets=np.array([0.0, 0.05, 0.15, 0.2]),
        rabi_rotations=np.array([1.57079633, 3.14159265, 3.14159265, 1.57079633]),
        azimuthal_angles=np.array([0.0, 0.0, 0.0, 0.0]),
        detuning_rotations=np.array([0.0, 0.0, 0.0, 0.0]),
        name=None,
    )

    driven_control = convert_dds_to_driven_control(
        dynamic_decoupling_sequence,
        maximum_rabi_rate=20.0 * np.pi,
        maximum_detuning_rate=2 * np.pi,
        name=None,
    )

    # There is no space for a gap between the pi/2-pulses and the adjacent pi-pulses,
    # so the resulting sequence should have 4 pulses + 1 gaps = 5 segments with non-zero duration
    assert sum(np.greater(driven_control.durations, 0.0)) == 5

    # ... of which 4 are X pulses (i.e. rabi_rotation > 0)
    assert sum(np.greater(driven_control.rabi_rates, 0.0)) == 4


def test_free_evolution_conversion():

    """
    Tests the conversion of free evolution.
    """
    _duration = 10.0
    _name = "test_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=np.array([]),
        rabi_rotations=np.array([]),
        azimuthal_angles=np.array([]),
        detuning_rotations=np.array([]),
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name,
    )

    _rabi_rates = np.array([0.0])
    _azimuthal_angles = np.array([0.0])
    _detunings = np.array([0.0])
    _durations = np.array([_duration])
    assert np.allclose(driven_control.rabi_rates, _rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, _azimuthal_angles)
    assert np.allclose(driven_control.detunings, _detunings)
    assert np.allclose(driven_control.durations, _durations)

    _duration = 10.0
    _name = "test_sequence"
    _offsets = np.array([0, _duration])
    _rabi_rotations = np.array([np.pi / 2, np.pi / 2])
    _azimuthal_angles = np.array([0, 0])
    _detuning_rotations = np.array([0, 0])

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name,
    )

    _rabi_rates = np.array([_maximum_rabi_rate, 0.0, _maximum_rabi_rate])
    _azimuthal_angles = np.array([0, 0, 0])
    _detunings = np.array([0, 0, 0])
    _durations = np.array([0.025, 9.95, 0.025])
    assert np.allclose(driven_control.rabi_rates, _rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, _azimuthal_angles)
    assert np.allclose(driven_control.detunings, _detunings)
    assert np.allclose(driven_control.durations, _durations)


def test_export_to_file():

    """
    Tests exporting to file.
    """
    _duration = 2.0
    _offsets = 2 * np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    _rabi_rotations = np.array([0.0, np.pi, 0.0, np.pi, 0.0])
    _azimuthal_angles = np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0])
    _detuning_rotations = np.array([0.0, 0.0, np.pi, 0.0, 0.0])
    _name = "test_sequence"

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name,
    )

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name,
    )

    _filename = "dds_qctrl_cylindrical.csv"
    driven_control.export_to_file(
        filename=_filename,
        file_format="Q-CTRL expanded",
        file_type="CSV",
        coordinates="cylindrical",
    )

    _filename = "dds_qctrl_cartesian.csv"
    driven_control.export_to_file(
        filename=_filename,
        file_format="Q-CTRL expanded",
        file_type="CSV",
        coordinates="cartesian",
    )

    _filename = "dds_qctrl_cylindrical.json"
    driven_control.export_to_file(
        filename=_filename,
        file_format="Q-CTRL expanded",
        file_type="JSON",
        coordinates="cylindrical",
    )

    _filename = "dds_qctrl_cartesian.json"
    driven_control.export_to_file(
        filename=_filename,
        file_format="Q-CTRL expanded",
        file_type="JSON",
        coordinates="cartesian",
    )

    _remove_file("dds_qctrl_cylindrical.csv")
    _remove_file("dds_qctrl_cartesian.csv")
    _remove_file("dds_qctrl_cylindrical.json")
    _remove_file("dds_qctrl_cartesian.json")


def _all_greater_or_close(array, value):
    """
    Returns True if array is greater or close to value, element-wise.
    """
    return np.all(
        np.logical_or(np.greater_equal(array, value), np.isclose(array, value))
    )
