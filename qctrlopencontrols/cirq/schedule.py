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
=============
cirq.schedule
=============
"""

import numpy as np

from ..dynamic_decoupling_sequences.dynamic_decoupling_sequence import DynamicDecouplingSequence
from ..exceptions.exceptions import ArgumentsValueError


def convert_dds_to_cirq_schedule(   #pylint: disable=too-many-locals
        dynamic_decoupling_sequence,
        target_qubits=None,
        gate_time=0.1,
        add_measurement=True,
        device=None):

    """Converts a Dynamic Decoupling Sequence into schedule
    as defined in cirq

    Parameters
    ----------
    dynamic_decoupling_sequence : DynamicDecouplingSequence
        The dynamic decoupling sequence
    target_qubits : list, optional
        List of target qubits for the sequence operation; the qubits must be
        cirq.Qid type; defaults to None in which case a 1-D lattice of one
        qubit is used (indexed as 0).
    gate_time : float, optional
        Time (in seconds) delay introduced by a gate; defaults to 0.1
    add_measurement : bool, optional
        If True, the schedule contains a measurement operation for each of the
        target qubits. Measurement from each of the qubits is associated
        with a string as key. The string is formatted as 'qubit-X' where
        X is a number between 0 and len(target_qubits).
    device : cirq.Device, optional
        A cirq.Device that specifies hardware constraints for validating operations.
        If None, a unconstrained device is used. See `Cirq Documentation
        <https://cirq.readthedocs.io/en/stable/schedules.html/>` _.

    Returns
    -------
    cirq.Schedule
        The schedule of sequence rotation operations.


    Raises
    ------
    ArgumentsValueError
        If any of the input parameters result in an invalid operation.

    Notes
    -----

    Dynamic Decoupling Sequences (DDS) consist of idealized pulse operation. Theoretically,
    these operations (pi-pulses in X,Y or Z) occur instantaneously. However, in practice,
    pulses require time. Therefore, this method of converting an idealized sequence
    results to a schedule that is only an approximate implementation of the idealized sequence.

    In idealized definition of DDS, `offsets` represents the instances within sequence
    `duration` where a pulse occurs instantaneously. A series of appropriate rotation
    operations is placed in order to represent these pulses.

    In cirq.schedule, the active pulses are scheduled to be activated at a certain
    instant calculated from the start of the sequence and continues for a duration
    of gate_time. This does not require identity gates to be placed between offsets.

    Q-CTRL Open Controls support operation resulting in rotation around at most one axis at
    any offset.
    """

    import cirq

    if dynamic_decoupling_sequence is None:
        raise ArgumentsValueError('No dynamic decoupling sequence provided.',
                                  {'dynamic_decoupling_sequence': dynamic_decoupling_sequence})

    if not isinstance(dynamic_decoupling_sequence, DynamicDecouplingSequence):
        raise ArgumentsValueError('Dynamical decoupling sequence is not recognized.'
                                  'Expected DynamicDecouplingSequence instance',
                                  {'type(dynamic_decoupling_sequence)':
                                       type(dynamic_decoupling_sequence)})

    if gate_time <= 0:
        raise ArgumentsValueError(
            'Time delay of gates must be greater than zero.',
            {'gate_time': gate_time})

    if target_qubits is None:
        target_qubits = [cirq.LineQubit(0)]

    if device is None:
        device = cirq.UnconstrainedDevice

    if not isinstance(device, cirq.Device):
        raise ArgumentsValueError('Device must be a cirq.Device type.',
                                  {'device': device})

    # time in nano seconds
    gate_time = gate_time * 1e9

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
    # offsets in nano seconds
    offsets = offsets * 1e9

    scheduled_operations = []
    offset_count = 0
    for op_idx in range(operations.shape[1]):

        rabi_rotation = dynamic_decoupling_sequence.rabi_rotations[offset_count]
        azimuthal_angle = dynamic_decoupling_sequence.azimuthal_angles[offset_count]
        x_rotation = rabi_rotation * np.cos(azimuthal_angle)
        y_rotation = rabi_rotation * np.sin(azimuthal_angle)
        z_rotation = dynamic_decoupling_sequence.detuning_rotations[offset_count]

        rotations = np.array([x_rotation, y_rotation, z_rotation])
        zero_pulses = np.isclose(rotations, 0.0).astype(np.int)
        nonzero_pulse_counts = 3 - np.sum(zero_pulses)
        if nonzero_pulse_counts > 1:
            raise ArgumentsValueError(
                'Open Controls support a sequence with one'
                'valid pulse at any offset. Found sequence'
                'with multiple rotation operations at an offset.',
                {'dynamic_decoupling_sequence': str(dynamic_decoupling_sequence),
                 'offset': dynamic_decoupling_sequence.offsets[op_idx],
                 'rabi_rotation': dynamic_decoupling_sequence.rabi_rotations[
                     op_idx],
                 'azimuthal_angle': dynamic_decoupling_sequence.azimuthal_angles[
                     op_idx],
                 'detuning_rotaion': dynamic_decoupling_sequence.detuning_rotations[
                     op_idx]}
            )

        for qubit in target_qubits:
            if nonzero_pulse_counts == 0:
                operation = cirq.ScheduledOperation(
                    time=cirq.Timestamp(nanos=offsets[op_idx]),
                    duration=cirq.Duration(nanos=gate_time),
                    operation=cirq.I(qubit))
            else:
                if not np.isclose(rotations[0], 0.0):
                    operation = cirq.ScheduledOperation(
                        time=cirq.Timestamp(nanos=offsets[op_idx]),
                        duration=cirq.Duration(nanos=gate_time),
                        operation=cirq.Rx(rotations[0])(qubit))
                elif not np.isclose(rotations[1], 0.0):
                    operation = cirq.ScheduledOperation(
                        time=cirq.Timestamp(nanos=offsets[op_idx]),
                        duration=cirq.Duration(nanos=gate_time),
                        operation=cirq.Rx(rotations[1])(qubit))
                elif not np.isclose(rotations[2], 0.):
                    operation = cirq.ScheduledOperation(
                        time=cirq.Timestamp(nanos=offsets[op_idx]),
                        duration=cirq.Duration(nanos=gate_time),
                        operation=cirq.Rx(rotations[2])(qubit))
            offset_count += 1
            scheduled_operations.append(operation)

    if add_measurement:
        for idx, qubit in enumerate(target_qubits):
            operation = cirq.ScheduledOperation(
                time=cirq.Timestamp(nanos=offsets[-1] + gate_time),
                duration=cirq.Duration(nanos=gate_time),
                operation=cirq.MeasurementGate(
                    1, key='qubit-{}'.format(idx))(qubit))
            scheduled_operations.append(operation)

    schedule = cirq.Schedule(device=device, scheduled_operations=scheduled_operations)
    return schedule
