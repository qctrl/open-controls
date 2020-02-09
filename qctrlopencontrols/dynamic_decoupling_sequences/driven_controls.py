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
=============================================
dynamic_decoupling_sequences.driven_controls
=============================================
"""

import numpy as np

from ..exceptions.exceptions import ArgumentsValueError
from ..driven_controls import (
    UPPER_BOUND_RABI_RATE, UPPER_BOUND_DETUNING_RATE)
from ..driven_controls.driven_control import DrivenControl


def _check_valid_operation(rabi_rotations, detuning_rotations):
    """
    Private method to check if there is a rabi_rotation and detuning rotation at the same
    offset

    Parameters
    ----------
    rabi_rotations : numpy.ndarray
        Rabi rotations at each offset
    detuning_rotations : numpy.ndarray
        Detuning rotations at each offset

    Returns
    -------
    bool
        Returns True if there is not an instance of rabi rotation and detuning rotation
        at the same offset
    """

    rabi_rotation_index = set(np.where(rabi_rotations > 0.)[0])
    detuning_rotation_index = set(np.where(detuning_rotations > 0.)[0])

    check_common_index = rabi_rotation_index.intersection(detuning_rotation_index)

    if check_common_index:
        return False

    return True


def _check_maximum_rotation_rate(
        maximum_rabi_rate, maximum_detuning_rate):
    """Checks if the maximum rabi and detuning rate are
    within valid limits

    Parameters
    ----------
    maximum_rabi_rate : float, optional
        Maximum Rabi Rate; Defaults to 1.0
    maximum_detuning_rate : float, optional
        Maximum Detuning Rate; Defaults to None

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid or a valid driven control cannot be
        created from the sequence parameters, maximum rabi rate and maximum detuning
        rate provided
    """

    # check against global parameters
    if maximum_rabi_rate < 0. or maximum_rabi_rate > UPPER_BOUND_RABI_RATE:
        raise ArgumentsValueError(
            'Maximum rabi rate must be between 0. and maximum value of {0}'.format(
                UPPER_BOUND_RABI_RATE),
            {'maximum_rabi_rate': maximum_rabi_rate},
            extras={'maximum_detuning_rate': maximum_detuning_rate,
                    'allowed_maximum_rabi_rate': UPPER_BOUND_RABI_RATE})

    if maximum_detuning_rate < 0. or maximum_detuning_rate > UPPER_BOUND_DETUNING_RATE:
        raise ArgumentsValueError(
            'Maximum detuning rate must be between 0. and maximum value of {0}'.format(
                UPPER_BOUND_DETUNING_RATE),
            {'maximum_detuning_rate': maximum_detuning_rate, },
            extras={'maximum_rabi_rate': maximum_rabi_rate,
                    'allowed_maximum_rabi_rate': UPPER_BOUND_RABI_RATE,
                    'allowed_maximum_detuning_rate': UPPER_BOUND_DETUNING_RATE})


def convert_dds_to_driven_control(
        dynamic_decoupling_sequence=None,
        maximum_rabi_rate=2*np.pi,
        maximum_detuning_rate=2*np.pi,
        **kwargs):
    """Creates a Driven Control based on the supplied DDS and
    other relevant information

    Parameters
    ----------
    dynamic_decoupling_sequence : qctrlopencontrols.DynamicDecouplingSequence
        The base DDS; Defaults to None
    maximum_rabi_rate : float, optional
        Maximum Rabi Rate; Defaults to 1.0
    maximum_detuning_rate : float, optional
        Maximum Detuning Rate; Defaults to None
    kwargs : dict, optional
        options to make the corresponding filter type.
        I.e. the options for primitive is described in doc for the PrimitivePulse class.

    Returns
    -------
    DrivenControls
        The Driven Control that contains the segments
        corresponding to the Dynamic Decoupling Sequence operation

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid or a valid driven control cannot be
        created from the sequence parameters, maximum rabi rate and maximum detuning
        rate provided

    Notes
    -----
    Driven pulse is defined as a sequence of control segments. Each segment performs
    an operation (rotation around one or more axes). While the dynamic decoupling
    sequence operation contains ideal instant operations, maximum rabi (detuning) rate
    defines a minimum time required to perform a given rotation operation. Therefore, each
    operation in sequence is converted to a flat-topped control segment with a finite duration.
    Each offset is taken as the mid-point of the control segment and the width of the
    segment is determined by (rotation/max_rabi(detuning)_rate).

    If the sequence contains operations at either of the extreme ends
    :math:`\\tau_0=0` and :math:`\\tau_{n+1}=\\tau`(duration of the sequence), there
    will be segments outside the boundary (segments starting before :math:`t<0`
    or finishing after the sequence duration :math:`t>\\tau`. In these cases, the segments
    on either of the extreme ends are shifted appropriately so that their start/end time
    falls entirely within the duration of the sequence.

    Moreover, a check is made to make sure the resulting control segments are non-overlapping.

    If appropriate control segments cannot be created, the conversion process raises
    an ArgumentsValueError.
    """

    if dynamic_decoupling_sequence is None:
        raise ArgumentsValueError('Dynamic decoupling sequence must be of '
                                  'DynamicDecoupling type.',
                                  {'type(dynamic_decoupling_sequence':
                                   type(dynamic_decoupling_sequence)})

    _check_maximum_rotation_rate(maximum_rabi_rate, maximum_detuning_rate)

    sequence_duration = dynamic_decoupling_sequence.duration
    offsets = dynamic_decoupling_sequence.offsets
    rabi_rotations = dynamic_decoupling_sequence.rabi_rotations
    azimuthal_angles = dynamic_decoupling_sequence.azimuthal_angles
    detuning_rotations = dynamic_decoupling_sequence.detuning_rotations

    # check for valid operation
    if not _check_valid_operation(rabi_rotations=rabi_rotations,
                                  detuning_rotations=detuning_rotations):
        raise ArgumentsValueError(
            'Sequence operation includes rabi rotation and '
            'detuning rotation at the same instance.',
            {'dynamic_decoupling_sequence': str(dynamic_decoupling_sequence)},
            extras={'maximum_rabi_rate': maximum_rabi_rate,
                    'maximum_detuning_rate': maximum_detuning_rate})

    # check if detuning rate is supplied if there is a detuning_rotation > 0
    if np.any(detuning_rotations > 0.) and maximum_detuning_rate is None:
        raise ArgumentsValueError(
            'Sequence operation includes detuning rotations. Please supply a valid '
            'maximum_detuning_rate.',
            {'detuning_rotations': dynamic_decoupling_sequence.detuning_rotations,
             'maximum_detuning_rate': maximum_detuning_rate},
            extras={'maximum_rabi_rate': maximum_rabi_rate})

    if offsets.size == 0:
        offsets = np.array([0, sequence_duration])
        rabi_rotations = np.array([0, 0])
        azimuthal_angles = np.array([0, 0])
        detuning_rotations = np.array([0, 0])

    if offsets[0] != 0:
        offsets = np.append([0], offsets)
        rabi_rotations = np.append([0], rabi_rotations)
        azimuthal_angles = np.append([0], azimuthal_angles)
        detuning_rotations = np.append([0], detuning_rotations)
    if offsets[-1] != sequence_duration:
        offsets = np.append(offsets, [sequence_duration])
        rabi_rotations = np.append(rabi_rotations, [0])
        azimuthal_angles = np.append(azimuthal_angles, [0])
        detuning_rotations = np.append(detuning_rotations, [0])

    offsets = offsets[np.newaxis, :]
    rabi_rotations = rabi_rotations[np.newaxis, :]
    azimuthal_angles = azimuthal_angles[np.newaxis, :]
    detuning_rotations = detuning_rotations[np.newaxis, :]

    operations = np.concatenate((offsets, rabi_rotations,
                                 azimuthal_angles, detuning_rotations),
                                axis=0)

    pulse_mid_points = operations[0, :]

    pulse_start_ends = np.zeros((
        operations.shape[1], 2))   # pylint: disable=unsubscriptable-object

    for op_idx in range(operations.shape[1]):   # pylint: disable=unsubscriptable-object

        if np.isclose(np.sum(operations[:, op_idx]), 0.0):
            continue

        if operations[3, op_idx] == 0: #no z_rotations
            if not np.isclose(operations[1, op_idx], 0.):
                half_pulse_duration = 0.5 * operations[1, op_idx] / maximum_rabi_rate
            else:
                half_pulse_duration = 0.5 * operations[2, op_idx] / maximum_rabi_rate

            pulse_start_ends[op_idx, 0] = pulse_mid_points[op_idx] - half_pulse_duration

            pulse_start_ends[op_idx, 1] = pulse_mid_points[op_idx] + half_pulse_duration
        else:
            pulse_start_ends[op_idx, 0] = pulse_mid_points[op_idx] - \
                                          0.5 * operations[3, op_idx] / maximum_detuning_rate

            pulse_start_ends[op_idx, 1] = pulse_mid_points[op_idx] + \
                                          0.5 * operations[3, op_idx] / maximum_detuning_rate

    # check if any of the pulses have gone outside the time limit [0, sequence_duration]
    # if yes, adjust the segment timing
    if pulse_start_ends[0, 0] < 0.:

        if np.sum(np.abs(pulse_start_ends[0, :])) == 0:
            pulse_start_ends[0, 0] = 0
        else:
            translation = 0. - (pulse_start_ends[0, 0])
            pulse_start_ends[0, :] = pulse_start_ends[0, :] + translation

    if pulse_start_ends[-1, 1] > sequence_duration:

        if np.sum(np.abs(pulse_start_ends[0, :])) == 2 * sequence_duration:
            pulse_start_ends[-1, 1] = sequence_duration
        else:
            translation = pulse_start_ends[-1, 1] - sequence_duration
            pulse_start_ends[-1, :] = pulse_start_ends[-1, :] - translation

    # four conditions to check
    # 1. Control segment start times should be monotonically increasing
    # 2. Control segment end times should be monotonically increasing
    # 3. Control segment start time must be less than its end time
    # 4. Adjacent segments should not be overlapping
    if (np.any(pulse_start_ends[0:-1, 0] - pulse_start_ends[1:, 0] > 0.) or
            np.any(pulse_start_ends[0:-1, 1] - pulse_start_ends[1:, 1] > 0.) or
            np.any(pulse_start_ends[:, 0] - pulse_start_ends[:, 1] > 0.) or
            np.any(pulse_start_ends[1:, 0]-pulse_start_ends[0:-1, 1] < 0.)):

        raise ArgumentsValueError('Pulse timing could not be properly deduced from '
                                  'the sequence operation offsets. Try increasing the '
                                  'maximum rabi rate or maximum detuning rate.',
                                  {'dynamic_decoupling_sequence': dynamic_decoupling_sequence,
                                   'maximum_rabi_rate': maximum_rabi_rate,
                                   'maximum_detuning_rate': maximum_detuning_rate},
                                  extras={'deduced_pulse_start_timing': pulse_start_ends[:, 0],
                                          'deduced_pulse_end_timing': pulse_start_ends[:, 1]})

    if np.allclose(pulse_start_ends, 0.0):
        # the original sequence should be a free evolution
        return DrivenControl(rabi_rates=[0.],
                             azimuthal_angles=[0.],
                             detunings=[0.],
                             durations=[sequence_duration],
                             **kwargs)

    control_rabi_rates = np.zeros((
        operations.shape[1]*2,))    # pylint: disable=unsubscriptable-object
    control_azimuthal_angles = np.zeros((
        operations.shape[1] * 2,))  # pylint: disable=unsubscriptable-object
    control_detunings = np.zeros((
        operations.shape[1] * 2,))  # pylint: disable=unsubscriptable-object
    control_durations = np.zeros((
        operations.shape[1] * 2,))  # pylint: disable=unsubscriptable-object

    pulse_segment_idx = 0
    for op_idx in range(0, operations.shape[1]):    # pylint: disable=unsubscriptable-object

        if operations[3, op_idx] == 0.0:
            control_rabi_rates[pulse_segment_idx] = maximum_rabi_rate
            control_azimuthal_angles[pulse_segment_idx] = operations[2, op_idx]
            control_durations[pulse_segment_idx] = (pulse_start_ends[op_idx, 1] -
                                                    pulse_start_ends[op_idx, 0])
        else:
            control_detunings[pulse_segment_idx] = operations[3, op_idx]
            control_durations[pulse_segment_idx] = (pulse_start_ends[op_idx, 1] -
                                                    pulse_start_ends[op_idx, 0])

        if op_idx != (operations.shape[1]-1):   # pylint: disable=unsubscriptable-object
            control_rabi_rates[pulse_segment_idx+1] = 0.
            control_azimuthal_angles[pulse_segment_idx+1] = 0.
            control_detunings[pulse_segment_idx+1] = 0.
            control_durations[pulse_segment_idx+1] = (pulse_start_ends[op_idx+1, 0] -
                                                      pulse_start_ends[op_idx, 1])

        pulse_segment_idx += 2

    # almost there; let us check if there is any segments with durations = 0
    control_rabi_rates = control_rabi_rates[control_durations > 0.]
    control_azimuthal_angles = control_azimuthal_angles[control_durations > 0.]
    control_detunings = control_detunings[control_durations > 0.]
    control_durations = control_durations[control_durations > 0.]

    return DrivenControl(rabi_rates=control_rabi_rates,
                         azimuthal_angles=control_azimuthal_angles,
                         detunings=control_detunings,
                         durations=control_durations,
                         **kwargs)


if __name__ == '__main__':
    pass
