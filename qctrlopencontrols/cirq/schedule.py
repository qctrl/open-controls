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

import cirq

from qctrlopencontrols.dynamic_decoupling_sequences import DynamicDecouplingSequence
from qctrlopencontrols.exceptions import ArgumentsValueError

from qctrlopencontrols.qiskit import get_rotations


def _get_cirq_schedule(dynamic_decoupling_sequence,
                       target_qubits,
                       gate_time,
                       add_measurement,
                       device):

    """Returns a scheduled operations constructed from dynamic decoupling sequence

    Parameters
    ----------
    dynamic_decoupling_sequence : DynamicDecouplingSequence
        The dynamic decoupling sequence
    target_qubits : list
        List of target qubits for the sequence operation; the qubits must be
        cirq.Qid type
    gate_time : float, optional
        Time (in seconds) delay introduced by a gate; defaults to 0.1
    add_measurement : bool
        If True, a measurement operation is added to each of the qubits.
    device : cirq.Device
        The device where these operations will be running.

    Returns
    -------
    cirq.Schedule
        The scheduled rotation operations. The Schedule object contains a
        series of desired gates at specific times measured from the start
        of the duration.

    Raises
    ------
    ArgumentsValueError
        If there is rotations around more than one axis at any of the offsets
    """

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
        instance_operation = np.array([dynamic_decoupling_sequence.rabi_rotations[op_idx],
                                       dynamic_decoupling_sequence.azimuthal_angles[op_idx],
                                       dynamic_decoupling_sequence.detuning_rotations[op_idx]
                                       ])

        rotations = get_rotations(instance_operation)
        nonzero_pulse_counts = 0
        for rotation in rotations:
            if not np.isclose(rotation, 0.0):
                nonzero_pulse_counts += 1
        if nonzero_pulse_counts > 1:
            raise ArgumentsValueError(
                'Open Controls support a sequence with one '
                'valid pulse at any offset. Found sequence '
                'with multiple rotation operations at an offset.',
                {'dynamic_decoupling_sequence': str(dynamic_decoupling_sequence),
                 'instance_operation': instance_operation})

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


def convert_dds_to_cirq_schedule(
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

    return _get_cirq_schedule(dynamic_decoupling_sequence=dynamic_decoupling_sequence,
                              target_qubits=target_qubits,
                              gate_time=gate_time,
                              add_measurement=add_measurement,
                              device=device)
