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
========================================
Tests for Dynamical Decoupling Sequences
========================================
"""
import os
import pytest
import numpy as np
from qctrlopencontrols.exceptions import ArgumentsValueError
from qctrlopencontrols import (
    DynamicDecouplingSequence, convert_dds_to_driven_controls)


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

    attributes_string = ','.join('{0}={1}'.format(attribute,
                                                  repr(getattr(sequence, attribute)))
                                 for attribute in sequence.base_attributes)
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

    _plot_times = np.array([0, 0, 0,
                            0.25, 0.25, 0.25,
                            0.5, 0.5, 0.5,
                            0.75, 0.75, 0.75,
                            1., 1., 1.])
    _plot_rabi_rotations = np.array([0, 0, 0,
                                     0, np.pi, 0,
                                     0, 0, 0,
                                     0, np.pi, 0,
                                     0, 0, 0])
    _plot_azimuthal_angles = np.array([0, 0, 0,
                                       0, 0, 0,
                                       0, 0, 0,
                                       0, 0, 0,
                                       0, 0, 0])

    _plot_detuning_rotations = np.array([0, 0, 0,
                                         0, 0, 0,
                                         0, 0, 0,
                                         0, 0, 0,
                                         0, 0, 0])

    seq = DynamicDecouplingSequence(
        duration=1.0,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angle,
        detuning_rotations=_detuning_rotations)

    plot_data = seq.get_plot_formatted_arrays()
    plot_rabi, plot_azimuthal, plot_detuning, plot_times = (
        plot_data['rabi_rotations'],
        plot_data['azimuthal_angles'],
        plot_data['detuning_rotations'],
        plot_data['times']
    )

    assert np.allclose(_plot_rabi_rotations, plot_rabi)
    assert np.allclose(_plot_azimuthal_angles, plot_azimuthal)
    assert np.allclose(_plot_detuning_rotations, plot_detuning)
    assert np.allclose(_plot_times, plot_times)

    # with both X and Y pi
    _offsets = np.array([0, 0.25, 0.5, 0.75, 1.00])
    _rabi_rotations = np.array([0, np.pi, 0, np.pi, 0])
    _azimuthal_angle = np.array([0, np.pi/2, 0, np.pi/2, 0])
    _detuning_rotations = np.array([0, 0, 0, 0, 0])

    _plot_rabi_rotations = np.array([0, 0, 0,
                                     0, np.pi, 0,
                                     0, 0, 0,
                                     0, np.pi, 0,
                                     0, 0, 0])
    _plot_azimuthal_angles = np.array([0, 0, 0,
                                       0, np.pi/2, 0,
                                       0, 0, 0,
                                       0, np.pi/2, 0,
                                       0, 0, 0])

    _plot_detuning_rotations = np.array([0, 0, 0,
                                         0, 0, 0,
                                         0, 0, 0,
                                         0, 0, 0,
                                         0, 0, 0])

    _plot_times = np.array([0, 0, 0,
                            0.25, 0.25, 0.25,
                            0.5, 0.5, 0.5,
                            0.75, 0.75, 0.75,
                            1., 1., 1.])
    seq = DynamicDecouplingSequence(
        duration=1.0,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angle,
        detuning_rotations=_detuning_rotations)

    plot_data = seq.get_plot_formatted_arrays()
    plot_rabi, plot_azimuthal, plot_detuning, plot_times = (
        plot_data['rabi_rotations'],
        plot_data['azimuthal_angles'],
        plot_data['detuning_rotations'],
        plot_data['times']
    )

    assert np.allclose(_plot_rabi_rotations, plot_rabi)
    assert np.allclose(_plot_azimuthal_angles, plot_azimuthal)
    assert np.allclose(_plot_detuning_rotations, plot_detuning)
    assert np.allclose(_plot_times, plot_times)


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
    driven_control = convert_dds_to_driven_controls(dd_sequence,
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
        [0., 0., 0., np.pi, 0., 0., 0]))
    assert np.allclose(driven_control.durations, np.array(
        [4.75e-1, 5e-2, 4.5e-1, 5e-2, 4.5e-1, 5e-2, 4.75e-1]))


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
    driven_control = convert_dds_to_driven_controls(
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
    driven_control = convert_dds_to_driven_controls(
        dd_sequence,
        maximum_rabi_rate=_maximum_rabi_rate,
        maximum_detuning_rate=_maximum_detuning_rate,
        name=_name)

    _rabi_rates = np.array([_maximum_rabi_rate, 0., _maximum_rabi_rate])
    _azimuthal_angles = np.array([0, 0, 0])
    _detunings = np.array([0, 0, 0])
    _durations = np.array([0.025, 9.95,  0.025])
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
    driven_control = convert_dds_to_driven_controls(dd_sequence,
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
