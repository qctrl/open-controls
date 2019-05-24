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
==============
pyquil.program
==============
"""

import numpy as np

from pyquil import Program
from pyquil.gates import I, RX, RY, RZ, MEASURE
from pyquil.quil import Pragma

from qctrlopencontrols.dynamic_decoupling_sequences import DynamicDecouplingSequence
from qctrlopencontrols.exceptions import ArgumentsValueError
from qctrlopencontrols.globals import (
    FIX_DURATION_UNITARY, INSTANT_UNITARY)


def convert_dds_to_program(
        dynamic_decoupling_sequence,
        target_qubits=None,
        gate_time=0.1,
        add_measurement=True,
        algorithm=INSTANT_UNITARY):

    """Converts a Dynamic Decoupling Sequence into quantum program
    as defined in Pyquil

    Parameters
    ----------
    dynamic_decoupling_sequence : DynamicDecouplingSequence
        The dynamic decoupling sequence
    target_qubits : list, optional
        List of integers specifying target qubits for the sequence operation;
        defaults to None
    gate_time : float, optional
        Time (in seconds) delay introduced by a gate; defaults to 0.1
    add_measurement : bool, optional
        If True, the circuit contains a measurement operation for each of the
        target qubits and a set of ClassicalRegister objects created with length
        equal to `len(target_qubits)`
    algorithm : str, optional
        One of 'fixed duration unitary' or 'instant unitary'; In the case of
        'fixed duration unitary', the sequence operations are assumed to be
        taking the amount of gate_time while 'instant unitary' assumes the sequence
        operations are instantaneous (and hence does not contribute to the delay between
        offsets). Defaults to 'instant unitary'.

    Returns
    -------
    pyquil.Program
        The Pyquil program containting gates specified by the rotations of
        dynamic decoupling sequence


    Raises
    ------
    ArgumentsValueError
        If any of the input parameters are invalid

    Notes
    -----

    Dynamic Decoupling Sequences (DDS) consist of idealized pulse operation. Theoretically,
    these operations (pi-pulses in X,Y or Z) occur instantaneously. However, in practice,
    pulses require time. Therefore, this method of converting an idealized sequence
    results to a circuit that is only an approximate implementation of the idealized sequence.

    In idealized definition of DDS, `offsets` represents the instances within sequence
    `duration` where a pulse occurs instantaneously. A series of appropriate gatges
    is placed in order to represent these pulses. The `gaps` or idle time in between active
    pulses are filled up with `identity` gates. Each identity gate introduces a delay of
    `gate_time`. In this implementation, the number of identity gates is determined by
    :math:`np.int(np.floor(offset_distance / gate_time))`. As a consequence, the duration of
    the real-circuit is :math:`gate_time \\times number_of_identity_gates +
    pulse_gate_time \\times number_of_pulses`.

    Q-CTRL Open Controls support operation resulting in rotation around at most one axis at
    any offset.
    """

    if dynamic_decoupling_sequence is None:
        raise ArgumentsValueError('No dynamic decoupling sequence provided.',
                                  {'dynamic_decoupling_sequence': dynamic_decoupling_sequence})

    if not isinstance(dynamic_decoupling_sequence, DynamicDecouplingSequence):
        raise ArgumentsValueError('Dynamical decoupling sequence is not recognized.'
                                  'Expected DynamicDecouplingSequence instance',
                                  {'type(dynamic_decoupling_sequence)':
                                       type(dynamic_decoupling_sequence)})

    if target_qubits is None:
        target_qubits = [0]

    if gate_time <= 0:
        raise ArgumentsValueError(
            'Time delay of identity gate must be greater than zero.',
            {'gate_time': gate_time})

    if np.any(target_qubits) < 0:
        raise ArgumentsValueError(
            'Every target qubits index must be positive.',
            {'target_qubits': target_qubits})

    if algorithm not in [FIX_DURATION_UNITARY, INSTANT_UNITARY]:
        raise ArgumentsValueError('Algorithm must be one of {} or {}'.format(
            INSTANT_UNITARY, FIX_DURATION_UNITARY), {'algorithm': algorithm})

    unitary_time = 0.
    if algorithm == FIX_DURATION_UNITARY:
        unitary_time = gate_time

    rabi_rotations = dynamic_decoupling_sequence.rabi_rotations
    azimuthal_angles = dynamic_decoupling_sequence.azimuthal_angles
    detuning_rotations = dynamic_decoupling_sequence.detuning_rotations

    if len(rabi_rotations.shape) == 1:
        rabi_rotations = rabi_rotations[np.newaxis, :]
    if len(azimuthal_angles.shape) == 1:
        azimuthal_angles = azimuthal_angles[np.newaxis, :]
    if len(detuning_rotations.shape) == 1:
        detuning_rotations = detuning_rotations[np.newaxis, :]

    operations = np.vstack((rabi_rotations, azimuthal_angles, detuning_rotations))
    offsets = dynamic_decoupling_sequence.offsets

    time_covered = 0
    program = Program()
    program += Pragma('PRESERVE_BLOCK')

    for operation_idx in range(operations.shape[1]):

        offset_distance = offsets[operation_idx] - time_covered

        if np.isclose(offset_distance, 0.0):
            offset_distance = 0.0

        if offset_distance < 0:
            raise ArgumentsValueError("Offsets cannot be placed properly",
                                      {'sequence_operations': operations})

        if offset_distance > 0:
            while (time_covered+gate_time) <= offsets[operation_idx]:
                for qubit in target_qubits:
                    program += I(qubit)
                time_covered += gate_time

        rabi_rotation = operations[0, operation_idx]
        azimuthal_angle = operations[1, operation_idx]
        x_rotation = rabi_rotation * np.cos(azimuthal_angle)
        y_rotation = rabi_rotation * np.sin(azimuthal_angle)
        z_rotation = operations[2, operation_idx]

        rotations = np.array([x_rotation, y_rotation, z_rotation])
        zero_pulses = np.isclose(rotations, 0.0).astype(np.int)
        nonzero_pulse_counts = 3 - np.sum(zero_pulses)
        if nonzero_pulse_counts > 1:
            raise ArgumentsValueError(
                'Open Controls support a sequence with one '
                'valid pulse at any offset. Found sequence '
                'with multiple rotation operations at an offset.',
                {'dynamic_decoupling_sequence': str(dynamic_decoupling_sequence),
                 'offset': dynamic_decoupling_sequence.offsets[operation_idx],
                 'rabi_rotation': dynamic_decoupling_sequence.rabi_rotations[
                     operation_idx],
                 'azimuthal_angle': dynamic_decoupling_sequence.azimuthal_angles[
                     operation_idx],
                 'detuning_rotaion': dynamic_decoupling_sequence.detuning_rotations[
                     operation_idx]}
            )

        for qubit in target_qubits:
            if nonzero_pulse_counts == 0:
                program += I(qubit)
            else:
                if not np.isclose(rotations[0], 0.0):
                    program += RX(rotations[0], qubit)
                elif not np.isclose(rotations[1], 0.0):
                    program += RY(rotations[1], qubit)
                elif not np.isclose(rotations[2], 0.):
                    program += RZ(rotations[2], qubit)

        if np.isclose(np.sum(rotations), 0.0):
            time_covered = offsets[operation_idx]
        else:
            time_covered = offsets[operation_idx] + unitary_time

    if add_measurement:
        for qubit in target_qubits:
            bit_name = 'qubit-{}'.format(qubit)
            measurement_bit = program.declare(bit_name, 'BIT', 1)
            program += MEASURE(qubit, measurement_bit)
    program += Pragma('END_PRESERVE_BLOCK')

    return program
