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
========================================
Tests for Dynamical Decoupling Sequences
========================================
"""
import os
import pytest
import numpy as np
from qctrlopencontrols.exceptions.exceptions import ArgumentsValueError
from qctrlopencontrols import (
    DynamicDecouplingSequence, convert_dds_to_driven_control)


def _remove_file(filename):
    """Removes the file after test done
    """

    if os.path.exists(filename):
        os.remove(filename)
    else:
        raise IOError('Could not find file {}'.format(
            filename))


def test_dynamical_decoupling_sequence():
    """Tests for the Dynamic Decoupling Sequence class
    """

    _duration = 2.
    _offsets = np.array([0.25, 0.5, 0.75, 1.00, 1.25, 1.50, 1.75])
    _rabi_rotations = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
    _azimthal_angles = 0.5*np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
    _detuning_rotations = np.array([0., 0., 0., 0., 0., 0., 0.])
    _name = 'test_sequence'

    sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name
    )

    assert sequence.duration == _duration
    assert np.allclose(sequence.offsets, _offsets)
    assert np.allclose(sequence.rabi_rotations, _rabi_rotations)
    assert np.allclose(sequence.azimuthal_angles, _azimthal_angles)
    assert np.allclose(sequence.detuning_rotations, _detuning_rotations)
    assert sequence.name == _name

    _repr_string = '{0.__class__.__name__!s}('.format(sequence)

    attributes = {
        'duration': sequence.duration,
        'offsets': sequence.offsets,
        'rabi_rotations': sequence.rabi_rotations,
        'azimuthal_angles': sequence.azimuthal_angles,
        'detuning_rotations': sequence.detuning_rotations,
        'name': sequence.name
    }

    attributes_string = ','.join('{0}={1}'.format(attribute,
                                                  repr(getattr(sequence, attribute)))
                                 for attribute in attributes)
    _repr_string += attributes_string
    _repr_string += ')'

    assert repr(sequence) == _repr_string

    _duration = 2.
    _offsets = np.array([0., 0.25, 0.5, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    _rabi_rotations = np.array([np.pi / 2, np.pi, np.pi, np.pi, np.pi, np.pi,
                                np.pi, np.pi, np.pi / 2])
    _azimthal_angles = 0.5 * np.array([0., np.pi, np.pi, np.pi, np.pi, np.pi,
                                       np.pi, np.pi, 0.])
    _detuning_rotations = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
    _name = 'test_sequence'

    sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name
    )

    assert sequence.duration == _duration
    assert np.allclose(sequence.offsets, _offsets)
    assert np.allclose(sequence.rabi_rotations, _rabi_rotations)
    assert np.allclose(sequence.azimuthal_angles, _azimthal_angles)
    assert np.allclose(sequence.detuning_rotations, _detuning_rotations)
    assert sequence.name == _name

    with pytest.raises(ArgumentsValueError):

        # not more than 10000 offsets
        _ = DynamicDecouplingSequence(
            duration=2.,
            offsets=2./2000*np.ones((20000, 1)),
            rabi_rotations=np.pi*np.ones((20000, 1))
        )

        # duration cannot be negative
        _ = DynamicDecouplingSequence(
            duration=-2.
        )


def test_sequence_plot():
    """
    Tests the plot data of sequences
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
        detuning_rotations=_detuning_rotations)

    plot_data = seq.export()

    _plot_rabi_offsets = [pulse['offset'] for pulse in plot_data['Rabi']]
    _plot_detuning_offsets = [pulse['offset'] for pulse in plot_data['Detuning']]
    _plot_rabi_rotations = [pulse['rotation'] for pulse in plot_data['Rabi']]
    _plot_detuning_rotations = [pulse['rotation'] for pulse in plot_data['Detuning']]

    assert np.allclose(_plot_rabi_offsets, _offsets)
    assert np.allclose(_plot_detuning_offsets, _offsets)

    assert np.allclose(np.abs(_plot_rabi_rotations), _rabi_rotations)
    assert np.allclose(np.angle(_plot_rabi_rotations), _azimuthal_angle)

    assert np.allclose(_plot_detuning_rotations, _detuning_rotations)

    # with both X and Y pi
    _offsets = np.array([0, 0.25, 0.5, 0.75, 1.00])
    _rabi_rotations = np.array([0, np.pi, 0, np.pi, 0])
    _azimuthal_angle = np.array([0, np.pi/2, 0, np.pi/2, 0])
    _detuning_rotations = np.array([0, 0, 0, 0, 0])

    seq = DynamicDecouplingSequence(
        duration=1.0,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angle,
        detuning_rotations=_detuning_rotations)

    plot_data = seq.export()

    _plot_rabi_offsets = [pulse['offset'] for pulse in plot_data['Rabi']]
    _plot_detuning_offsets = [pulse['offset'] for pulse in plot_data['Detuning']]
    _plot_rabi_rotations = [pulse['rotation'] for pulse in plot_data['Rabi']]
    _plot_detuning_rotations = [pulse['rotation'] for pulse in plot_data['Detuning']]

    assert np.allclose(_plot_rabi_offsets, _offsets)
    assert np.allclose(_plot_detuning_offsets, _offsets)

    assert np.allclose(np.abs(_plot_rabi_rotations), _rabi_rotations)
    assert np.allclose(np.angle(_plot_rabi_rotations), _azimuthal_angle)

    assert np.allclose(_plot_detuning_rotations, _detuning_rotations)


def test_pretty_string_format():

    """Tests __str__ of the dynamic decoupling sequence
    """

    _duration = 1.
    _offsets = np.array([0., 0.5, 1.])
    _rabi_rotations = np.array([0., np.pi, 0.])
    _azimuthal_angles = 0.5 * np.array([0., np.pi/2, np.pi])
    _detuning_rotations = np.array([np.pi/4, 0., 0.])
    _name = 'test_sequence'

    dd_sequence = DynamicDecouplingSequence(
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _pretty_string = ['test_sequence:']
    _pretty_string.append('Duration = {}'.format(_duration))
    _pretty_string.append('Offsets = [{},{},{}] x {}'.format(_offsets[0],
                                                             _offsets[1],
                                                             _offsets[2],
                                                             _duration))
    _pretty_string.append('Rabi Rotations = [{},{},{}] x pi'.format(
        _rabi_rotations[0]/np.pi, _rabi_rotations[1]/np.pi, _rabi_rotations[2]/np.pi))
    _pretty_string.append('Azimuthal Angles = [{},{},{}] x pi'.format(
        _azimuthal_angles[0]/np.pi, _azimuthal_angles[1]/np.pi, _azimuthal_angles[2]/np.pi))
    _pretty_string.append('Detuning Rotations = [{},{},{}] x pi'.format(
        _detuning_rotations[0] / np.pi, _detuning_rotations[1] / np.pi,
        _detuning_rotations[2] / np.pi))

    _pretty_string = '\n'.join(_pretty_string)

    assert _pretty_string == str(dd_sequence)

    dd_sequence = DynamicDecouplingSequence(
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations)

    _pretty_string = list()
    _pretty_string.append('Duration = {}'.format(_duration))
    _pretty_string.append('Offsets = [{},{},{}] x {}'.format(
        _offsets[0],
        _offsets[1],
        _offsets[2],
        _duration))
    _pretty_string.append('Rabi Rotations = [{},{},{}] x pi'.format(
        _rabi_rotations[0] / np.pi, _rabi_rotations[1] / np.pi, _rabi_rotations[2] / np.pi))
    _pretty_string.append('Azimuthal Angles = [{},{},{}] x pi'.format(
        _azimuthal_angles[0] / np.pi, _azimuthal_angles[1] / np.pi, _azimuthal_angles[2] / np.pi))
    _pretty_string.append('Detuning Rotations = [{},{},{}] x pi'.format(
        _detuning_rotations[0] / np.pi, _detuning_rotations[1] / np.pi,
        _detuning_rotations[2] / np.pi))
    _pretty_string = '\n'.join(_pretty_string)

    assert _pretty_string == str(dd_sequence)


def test_conversion_to_driven_controls():

    """Tests the method to convert a DDS to Driven Control
    """
    _duration = 2.
    _offsets = 2*np.array([0.25, 0.5, 0.75])
    _rabi_rotations = np.array([np.pi, 0., np.pi])
    _azimuthal_angles = np.array([np.pi / 2, 0., 0.])
    _detuning_rotations = np.array([0., np.pi, 0.])
    _name = 'test_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20*np.pi
    _maximum_detuning_rate = 20*np.pi
    driven_control = convert_dds_to_driven_control(dd_sequence,
                                                   maximum_rabi_rate=_maximum_rabi_rate,
                                                   maximum_detuning_rate=_maximum_detuning_rate,
                                                   name=_name)

    assert np.allclose(driven_control.rabi_rates, np.array(
        [0., _maximum_rabi_rate, 0., 0., 0.,
         _maximum_rabi_rate, 0.]))
    assert np.allclose(driven_control.azimuthal_angles, np.array(
        [0., _azimuthal_angles[0], 0., 0., 0.,
         _azimuthal_angles[2], 0.]))
    assert np.allclose(driven_control.detunings, np.array(
        [0., 0., 0., _maximum_detuning_rate, 0., 0., 0]))
    assert np.allclose(driven_control.durations, np.array(
        [4.75e-1, 5e-2, 4.5e-1, 5e-2, 4.5e-1, 5e-2, 4.75e-1]))


def test_conversion_of_pi_2_pulses_to_driven_controls():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    pi/2-pulses in the x, y, and z directions.
    """
    _duration = 6.
    _offsets = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    _rabi_rotations = np.array([np.pi/2, 0., np.pi/2, np.pi/2, 0., np.pi/2])
    _azimuthal_angles = np.array([np.pi / 2, 0., 0., -np.pi/2, 0, np.pi])
    _detuning_rotations = np.array([0., np.pi/2, 0., 0., -np.pi/2, 0])
    _name = 'pi2_pulse_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20*np.pi
    _maximum_detuning_rate = 20*np.pi
    driven_control = convert_dds_to_driven_control(dd_sequence,
                                                   maximum_rabi_rate=_maximum_rabi_rate,
                                                   maximum_detuning_rate=_maximum_detuning_rate,
                                                   name=_name)

    assert np.allclose(driven_control.rabi_rates, np.array(
        [0., _maximum_rabi_rate, 0., 0., 0., _maximum_rabi_rate,
         0., _maximum_rabi_rate, 0., 0., 0., _maximum_rabi_rate, 0.]))
    assert np.allclose(driven_control.azimuthal_angles, np.array(
        [0., _azimuthal_angles[0], 0., 0., 0., _azimuthal_angles[2],
         0., _azimuthal_angles[3], 0., 0., 0., _azimuthal_angles[5], 0.]))
    assert np.allclose(driven_control.detunings, np.array(
        [0., 0., 0., _maximum_detuning_rate, 0., 0.,
         0., 0., 0., -_maximum_detuning_rate, 0., 0., 0.]))
    assert np.allclose(driven_control.durations, np.array(
        [4.875e-1, 2.5e-2, 9.75e-1, 2.5e-2, 9.75e-1, 2.5e-2,
         9.75e-1, 2.5e-2, 9.75e-1, 2.5e-2, 9.75e-1, 2.5e-2, 4.875e-1]))


def test_conversion_of_x_pi_2_pulses_at_extremities():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    X pi/2-pulses in inverse directions, in the beginning and end of the sequence.
    """
    _duration = 1.
    _offsets = np.array([0., _duration])
    _rabi_rotations = np.array([np.pi/2, np.pi/2])
    _azimuthal_angles = np.array([np.pi, 0])
    _detuning_rotations = np.array([0., 0])
    _name = 'x_pi2_pulse_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20*np.pi
    _maximum_detuning_rate = 20*np.pi
    driven_control = convert_dds_to_driven_control(dd_sequence,
                                                   maximum_rabi_rate=_maximum_rabi_rate,
                                                   maximum_detuning_rate=_maximum_detuning_rate,
                                                   name=_name)

    assert np.allclose(driven_control.rabi_rates, np.array(
        [_maximum_rabi_rate, 0., _maximum_rabi_rate]))
    assert np.allclose(driven_control.azimuthal_angles, np.array(
        [_azimuthal_angles[0], 0., _azimuthal_angles[1]]))
    assert np.allclose(driven_control.detunings, np.array(
        [0., 0., 0.]))
    assert np.allclose(driven_control.durations, np.array(
        [2.5e-2, _duration - 2* 2.5e-2, 2.5e-2]))


def test_conversion_of_y_pi_2_pulses_at_extremities():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    Y pi/2-pulses in inverse directions, in the beginning and end of the sequence.
    """
    _duration = 1.
    _offsets = np.array([0., _duration])
    _rabi_rotations = np.array([np.pi/2, np.pi/2])
    _azimuthal_angles = np.array([-np.pi/2, np.pi/2])
    _detuning_rotations = np.array([0., 0])
    _name = 'y_pi2_pulse_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20*np.pi
    _maximum_detuning_rate = 20*np.pi
    driven_control = convert_dds_to_driven_control(dd_sequence,
                                                   maximum_rabi_rate=_maximum_rabi_rate,
                                                   maximum_detuning_rate=_maximum_detuning_rate,
                                                   name=_name)

    assert np.allclose(driven_control.rabi_rates, np.array(
        [_maximum_rabi_rate, 0., _maximum_rabi_rate]))
    assert np.allclose(driven_control.azimuthal_angles, np.array(
        [_azimuthal_angles[0], 0., _azimuthal_angles[1]]))
    assert np.allclose(driven_control.detunings, np.array(
        [0., 0., 0.]))
    assert np.allclose(driven_control.durations, np.array(
        [2.5e-2, _duration - 2* 2.5e-2, 2.5e-2]))



def test_conversion_of_z_pi_2_pulses_at_extremities():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    Z pi/2-pulses in inverse directions, in the beginning and end of the sequence.
    """
    _duration = 1.
    _offsets = np.array([0., _duration])
    _rabi_rotations = np.array([0, 0])
    _azimuthal_angles = np.array([0, 0])
    _detuning_rotations = np.array([-np.pi/2, np.pi/2])
    _name = 'z_pi2_pulse_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20*np.pi
    _maximum_detuning_rate = 20*np.pi
    driven_control = convert_dds_to_driven_control(dd_sequence,
                                                   maximum_rabi_rate=_maximum_rabi_rate,
                                                   maximum_detuning_rate=_maximum_detuning_rate,
                                                   name=_name)

    assert np.allclose(driven_control.rabi_rates, np.array(
        [0, 0., 0]))
    assert np.allclose(driven_control.azimuthal_angles, np.array(
        [0, 0., 0]))
    assert np.allclose(driven_control.detunings, np.array(
        [-_maximum_detuning_rate, 0., _maximum_detuning_rate]))
    assert np.allclose(driven_control.durations, np.array(
        [2.5e-2, _duration - 2* 2.5e-2, 2.5e-2]))



def test_conversion_of_pulses_with_arbitrary_rabi_rotations():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    Y pulses with rabi rotations that assume arbitrary values between 0 and pi.
    """
    _number_of_pulses = 30
    _duration = _number_of_pulses * 1.
    _offsets = np.linspace(0.5, _duration-0.5, _number_of_pulses)
    _rabi_rotations = np.pi/(_number_of_pulses+1)*np.linspace(1.,
                                                              _number_of_pulses,
                                                              _number_of_pulses)
    _azimuthal_angles = np.repeat(np.pi/2., _number_of_pulses)
    _detuning_rotations = np.repeat(0, _number_of_pulses)
    _name = 'arbitrary_rabi_rotation_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20*np.pi
    _maximum_detuning_rate = 10*np.pi
    minimum_segment_duration = np.mean(_rabi_rotations/_maximum_rabi_rate)

    driven_control = convert_dds_to_driven_control(dd_sequence,
                                                   maximum_rabi_rate=_maximum_rabi_rate,
                                                   maximum_detuning_rate=_maximum_detuning_rate,
                                                   minimum_segment_duration=minimum_segment_duration,
                                                   name=_name)

    expected_rabi_rates = np.zeros(2*_number_of_pulses+1)

    expected_azimuthal_angles = np.zeros(2*_number_of_pulses+1)
    expected_azimuthal_angles[1::2] = _azimuthal_angles

    expected_detuning_rates = np.zeros(2*_number_of_pulses+1)

    pulse_durations = np.maximum(_rabi_rotations/_maximum_rabi_rate,
                                 minimum_segment_duration)
    expected_rabi_rates[1::2] = _rabi_rotations/pulse_durations

    expected_durations = np.zeros(2*_number_of_pulses+1)
    expected_durations[1::2] = pulse_durations
    # durations of the middle gaps are: 1 - half of each of the neighboring pulses
    expected_durations[2:-2:2] = 1. - 0.5*pulse_durations[:-1] - 0.5*pulse_durations[1:]
    # initial and final gaps are just half of that
    expected_durations[0] = 0.5 - 0.5*pulse_durations[0]
    expected_durations[-1] = 0.5 - 0.5*pulse_durations[-1]

    assert np.allclose(driven_control.rabi_rates, expected_rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, expected_azimuthal_angles)
    assert np.allclose(driven_control.detunings, expected_detuning_rates)
    assert np.allclose(driven_control.durations, expected_durations)



def test_conversion_of_pulses_with_arbitrary_azimuthal_angles():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    pi-pulses with azimuthal angles that assume arbitrary values between 0 and pi/2.
    """
    _number_of_pulses = 30
    _duration = _number_of_pulses * 1.
    _offsets = np.linspace(0.5, _duration-0.5, _number_of_pulses)
    _rabi_rotations = np.repeat(np.pi, _number_of_pulses)
    _azimuthal_angles = (np.pi/2)/(_number_of_pulses+1)*np.linspace(1.,
                                                                    _number_of_pulses,
                                                                    _number_of_pulses)
    _detuning_rotations = np.repeat(0, _number_of_pulses)
    _name = 'arbitrary_azimuthal_angle_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20*np.pi
    _maximum_detuning_rate = 10*np.pi
    minimum_segment_duration = np.mean(_rabi_rotations/_maximum_rabi_rate)

    driven_control = convert_dds_to_driven_control(dd_sequence,
                                                   maximum_rabi_rate=_maximum_rabi_rate,
                                                   maximum_detuning_rate=_maximum_detuning_rate,
                                                   minimum_segment_duration=minimum_segment_duration,
                                                   name=_name)

    expected_rabi_rates = np.zeros(2*_number_of_pulses+1)

    expected_azimuthal_angles = np.zeros(2*_number_of_pulses+1)
    expected_azimuthal_angles[1::2] = _azimuthal_angles

    expected_detuning_rates = np.zeros(2*_number_of_pulses+1)

    pulse_durations = np.maximum(_rabi_rotations/_maximum_rabi_rate,
                                 minimum_segment_duration)
    expected_rabi_rates[1::2] = _rabi_rotations/pulse_durations

    expected_durations = np.zeros(2*_number_of_pulses+1)
    expected_durations[1::2] = pulse_durations
    # durations of the middle gaps are: 1 - half of each of the neighboring pulses
    expected_durations[2:-2:2] = 1. - 0.5*pulse_durations[:-1] - 0.5*pulse_durations[1:]
    # initial and final gaps are just half of that
    expected_durations[0] = 0.5 - 0.5*pulse_durations[0]
    expected_durations[-1] = 0.5 - 0.5*pulse_durations[-1]

    assert np.allclose(driven_control.rabi_rates, expected_rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, expected_azimuthal_angles)
    assert np.allclose(driven_control.detunings, expected_detuning_rates)
    assert np.allclose(driven_control.durations, expected_durations)



def test_conversion_of_pulses_with_arbitrary_detuning_rotations():
    """
    Tests if the method to convert a DDS to driven controls handles properly
    Z pulses with detuning rotations that assume arbitrary values between 0 and pi.
    """
    _number_of_pulses = 10
    _duration = _number_of_pulses * 1.
    _offsets = np.linspace(0.5, _duration-0.5, _number_of_pulses)
    _rabi_rotations = np.repeat(0., _number_of_pulses)
    _azimuthal_angles = np.repeat(0., _number_of_pulses)
    _detuning_rotations = np.pi/(_number_of_pulses+1)*np.linspace(1.,
                                                                  _number_of_pulses,
                                                                  _number_of_pulses)
    _name = 'arbitrary_detuning_rotation_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20*np.pi
    _maximum_detuning_rate = 10*np.pi
    minimum_segment_duration = np.mean(_detuning_rotations/_maximum_detuning_rate)

    driven_control = convert_dds_to_driven_control(dd_sequence,
                                                   maximum_rabi_rate=_maximum_rabi_rate,
                                                   maximum_detuning_rate=_maximum_detuning_rate,
                                                   minimum_segment_duration=minimum_segment_duration,
                                                   name=_name)

    expected_rabi_rates = np.zeros(2*_number_of_pulses+1)

    expected_azimuthal_angles = np.zeros(2*_number_of_pulses+1)
    expected_azimuthal_angles[1::2] = _azimuthal_angles

    expected_detuning_rates = np.zeros(2*_number_of_pulses+1)

    pulse_durations = np.maximum(minimum_segment_duration,
                                 _detuning_rotations/_maximum_detuning_rate)
    expected_detuning_rates[1::2] = _detuning_rotations/pulse_durations

    expected_durations = np.zeros(2*_number_of_pulses+1)
    expected_durations[1::2] = pulse_durations
    # durations of the middle gaps are: 1 - half of each of the neighboring pulses
    expected_durations[2:-2:2] = 1. - 0.5*pulse_durations[:-1] - 0.5*pulse_durations[1:]
    # initial and final gaps are just half of that
    expected_durations[0] = 0.5 - 0.5*pulse_durations[0]
    expected_durations[-1] = 0.5 - 0.5*pulse_durations[-1]

    print (driven_control.durations)
    print (driven_control.detunings)
    assert np.allclose(driven_control.rabi_rates, expected_rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, expected_azimuthal_angles)
    assert np.allclose(driven_control.detunings, expected_detuning_rates)
    assert np.allclose(driven_control.durations, expected_durations)


def test_free_evolution_conversion():

    """Tests the conversion of free evolution
    """
    _duration = 10.
    _name = 'test_sequence'
    _offsets = []
    _rabi_rotations = []
    _azimuthal_angles = []
    _detuning_rotations = []

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name)

    _rabi_rates = np.array([0.])
    _azimuthal_angles = np.array([0.])
    _detunings = np.array([0.])
    _durations = np.array([_duration])
    assert np.allclose(driven_control.rabi_rates, _rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, _azimuthal_angles)
    assert np.allclose(driven_control.detunings, _detunings)
    assert np.allclose(driven_control.durations, _durations)

    _duration = 10.
    _name = 'test_sequence'
    _offsets = [0, _duration]
    _rabi_rotations = [np.pi/2, np.pi/2]
    _azimuthal_angles = [0, 0]
    _detuning_rotations = [0, 0]

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name)

    _rabi_rates = np.array([_maximum_rabi_rate, 0., _maximum_rabi_rate])
    _azimuthal_angles = np.array([0, 0, 0])
    _detunings = np.array([0, 0, 0])
    _durations = np.array([0.025, 9.95, 0.025])
    assert np.allclose(driven_control.rabi_rates, _rabi_rates)
    assert np.allclose(driven_control.azimuthal_angles, _azimuthal_angles)
    assert np.allclose(driven_control.detunings, _detunings)
    assert np.allclose(driven_control.durations, _durations)


def test_export_to_file():

    """Tests exporting to file
    """
    _duration = 2.
    _offsets = 2 * np.array([0., 0.25, 0.5, 0.75, 1.])
    _rabi_rotations = np.array([0., np.pi, 0., np.pi, 0.])
    _azimuthal_angles = np.array([0., np.pi / 2, 0., 0., 0.])
    _detuning_rotations = np.array([0., 0., np.pi, 0., 0.])
    _name = 'test_sequence'

    dd_sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations,
        name=_name)

    _maximum_rabi_rate = 20 * np.pi
    _maximum_detuning_rate = 20 * np.pi
    driven_control = convert_dds_to_driven_control(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name)

    _filename = 'dds_qctrl_cylindrical.csv'
    driven_control.export_to_file(
        filename=_filename,
        file_format='Q-CTRL expanded',
        file_type='CSV',
        coordinates='cylindrical'
    )

    _filename = 'dds_qctrl_cartesian.csv'
    driven_control.export_to_file(
        filename=_filename,
        file_format='Q-CTRL expanded',
        file_type='CSV',
        coordinates='cartesian'
    )

    _filename = 'dds_qctrl_cylindrical.json'
    driven_control.export_to_file(
        filename=_filename,
        file_format='Q-CTRL expanded',
        file_type='JSON',
        coordinates='cylindrical'
    )

    _filename = 'dds_qctrl_cartesian.json'
    driven_control.export_to_file(
        filename=_filename,
        file_format='Q-CTRL expanded',
        file_type='JSON',
        coordinates='cartesian'
    )

    _remove_file('dds_qctrl_cylindrical.csv')
    _remove_file('dds_qctrl_cartesian.csv')
    _remove_file('dds_qctrl_cylindrical.json')
    _remove_file('dds_qctrl_cartesian.json')


if __name__ == '__main__':
    pass
