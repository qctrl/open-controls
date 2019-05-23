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
=================
cirq.cirq_circuit
=================
"""

import numpy as np

import cirq

from qctrlopencontrols.dynamic_decoupling_sequences import DynamicDecouplingSequence
from qctrlopencontrols.exceptions import ArgumentsValueError

from .constants import (SCHEDULED_CIRCUIT, STANDARD_CIRCUIT,
                        DEFAULT_PRE_POST_ROTATION_MATRIX)


def _get_circuit_gate_list(dynamic_decoupling_sequence,
                           gate_time,
                           unitary_time):

    """Converts the operations in a sequence into list of gates
    of a circuit

    Parameters
    ----------
    dynamic_decoupling_sequence : DynamicDecouplingSequence
        Dynamic decoupling sequence instance
    gate_time : float
        Indicates the delay time of the identity gates
    unitary_time : float
        Indicates the delay time introduced by unitary gates

    Returns
    -------
    list
        A list of circuit components with required parameters

    Raises
    ------
    ArgumentsValueError
        If the offsets cannot be placed properly
    """

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
    circuit_operations = []
    for operation_idx in range(operations.shape[1]):

        offset_distance = offsets[operation_idx] - time_covered

        if np.isclose(offset_distance, 0.0):
            offset_distance = 0.0

        if offset_distance < 0:
            raise ArgumentsValueError("Offsets cannot be placed properly",
                                      {'sequence_operations': operations})
        if offset_distance == 0:
            circuit_operations.append('offset')
            if np.isclose(np.sum(operations[:, operation_idx]), 0.0):
                time_covered = offsets[operation_idx]
            else:
                time_covered = offsets[operation_idx] + unitary_time
        else:
            number_of_id_gates = 0
            while (time_covered + (number_of_id_gates+1) * gate_time) <= \
                    offsets[operation_idx]:
                circuit_operations.append('id')
                number_of_id_gates += 1
            circuit_operations.append('offset')
            time_covered = offsets[operation_idx] + unitary_time

    return circuit_operations


def _get_rotations(operation):

    """Returns the pulses based of the rotation operation

    Parameters
    ----------
    operation : numpy.ndarray
        1-D array (length=3) consisting of rabi rotation, azimuthal_angle
        and detuning_rotation at an offset of a sequence

    Returns
    -------
    numpy.ndarray
        A 1-D array of length 3 containing x_rotation, y_rotation and z-rotation
        calculate from sequence operation
    """

    x_rotation = operation[0] * np.cos(operation[1])
    y_rotation = operation[0] * np.sin(operation[1])
    z_rotation = operation[2]

    pulses = np.array([x_rotation, y_rotation, z_rotation])

    return pulses


def _get_scheduled_circuit(dynamic_decoupling_sequence,
                           target_qubits,
                           gate_time,
                           pre_post_gate,
                           add_measurement,
                           device):

    """Returns a scheduled circuit operation constructed from
    dynamic decoupling sequence

    Parameters
    ----------
    dynamic_decoupling_sequence : DynamicDecouplingSequence
        The dynamic decoupling sequence
    target_qubits : list
        List of target qubits for the sequence operation; the qubits must be
        cirq.Qid type
    gate_time : float, optional
        Time (in seconds) delay introduced by a gate; defaults to 0.1
    pre_post_gate : SingleQubitGate
        A SingleQubitGate type (defined in cirq package) defined by a 2x2
        unitary matrix.
    add_measurement : bool
        If True, a measurement operation is added to each of the qubits.
    device : cirq.Device
        The device where these operations will be running.

    Returns
    -------
    cirq.Schedule
        The scheduled circuit operations. The Schedule object contains a
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

    circuit_operations = []
    if pre_post_gate is not None:
        for qubit in target_qubits:
            operation = cirq.ScheduledOperation(
                time=cirq.Timestamp(nanos=0),
                duration=cirq.Duration(nanos=gate_time),
                operation=pre_post_gate(qubit))
            circuit_operations.append(operation)
        offsets = offsets + gate_time

    offset_count = 0
    for op_idx in range(operations.shape[1]):
        instance_operation = np.array([dynamic_decoupling_sequence.rabi_rotations[op_idx],
                                       dynamic_decoupling_sequence.azimuthal_angles[op_idx],
                                       dynamic_decoupling_sequence.detuning_rotations[op_idx]
                                       ])

        rotations = _get_rotations(instance_operation)
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
            circuit_operations.append(operation)

    if pre_post_gate is not None:
        for qubit in target_qubits:
            operation = cirq.ScheduledOperation(
                time=cirq.Timestamp(nanos=offsets[-1] + gate_time),
                duration=cirq.Duration(nanos=gate_time),
                operation=pre_post_gate(qubit))
            circuit_operations.append(operation)
        offsets = offsets + gate_time

    if add_measurement:
        for idx, qubit in enumerate(target_qubits):
            operation = cirq.ScheduledOperation(
                time=cirq.Timestamp(nanos=offsets[-1] + gate_time),
                duration=cirq.Duration(nanos=gate_time),
                operation=cirq.MeasurementGate(
                    1, key='qubit-{}'.format(idx))(qubit))
            circuit_operations.append(operation)

    schedule = cirq.Schedule(device=device, scheduled_operations=circuit_operations)
    return schedule


def _get_standard_circuit(dynamic_decoupling_sequence,
                          target_qubits,
                          gate_time,
                          pre_post_gate,
                          add_measurement):

    """Returns a standard circuit constructed from dynamic
    decoupling sequence

    Parameters
    ----------
    dynamic_decoupling_sequence : DynamicDecouplingSequence
        The dynamic decoupling sequence
    target_qubits : list
        List of target qubits for the sequence operation; the qubits must be
        cirq.Qid type
    gate_time : float, optional
        Time (in seconds) delay introduced by a gate; defaults to 0.1
    pre_post_gate : SingleQubitGate
        A SingleQubitGate type (defined in cirq package) defined by a 2x2
        unitary matrix.
    add_measurement : bool
        If True, a measurement operation is added to each of the qubits.

    Returns
    -------
    cirq.Circuit
        The circuit prepared from dynamic decoupling sequence. In standard circuit
        the desired decoupling pulses are placed at offsets and the duration between
        the pulses are constructed from identity gates with delays equal to 'gate_time'.

    Raises
    ------
    ArgumentsValueError
        If there is rotations around more than one axis at any of the offsets
    """

    unitary_time = gate_time
    circuit_gate_list = _get_circuit_gate_list(
        dynamic_decoupling_sequence=dynamic_decoupling_sequence,
        gate_time=gate_time,
        unitary_time=unitary_time)

    circuit = cirq.Circuit()

    if pre_post_gate is not None:
        gate_list = []
        for qubit in target_qubits:
            gate_list.append(pre_post_gate(qubit))
        circuit.append(gate_list)

    offset_count = 0
    for gate in circuit_gate_list:

        if gate == 'id':
            gate_list = []
            for qubit in target_qubits:
                gate_list.append(cirq.I(qubit))
            circuit.append(gate_list)
            continue

        instance_operation = np.array(
            [dynamic_decoupling_sequence.rabi_rotations[offset_count],
             dynamic_decoupling_sequence.azimuthal_angles[offset_count],
             dynamic_decoupling_sequence.detuning_rotations[offset_count]])

        rotations = _get_rotations(instance_operation)
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
        gate_list = []
        for qubit in target_qubits:
            if nonzero_pulse_counts == 0:
                gate_list.append(cirq.I(qubit))
            else:
                if not np.isclose(rotations[0], 0.0):
                    gate_list.append(cirq.Rx(rotations[0])(qubit))
                elif not np.isclose(rotations[1], 0.0):
                    gate_list.append(cirq.Ry(rotations[1])(qubit))
                elif not np.isclose(rotations[2], 0.):
                    gate_list.append(cirq.Rz(rotations[2])(qubit))
        offset_count += 1
        circuit.append(gate_list)

    if pre_post_gate is not None:
        gate_list = []
        for qubit in target_qubits:
            gate_list.append(pre_post_gate(qubit))
        circuit.append(gate_list)

    if add_measurement:
        gate_list = []
        for idx, qubit in enumerate(target_qubits):
            gate_list.append(cirq.measure(qubit, key='qubit-{}'.format(idx)))
        circuit.append(gate_list)

    return circuit


def convert_dds_to_cirq_circuit(
        dynamic_decoupling_sequence,
        target_qubits=None,
        gate_time=0.1,
        pre_post_gate_unitary_matrix=DEFAULT_PRE_POST_ROTATION_MATRIX,
        add_measurement=True,
        circuit_type=STANDARD_CIRCUIT,
        device=None):

    """Converts a Dynamic Decoupling Sequence into quantum circuit
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
    pre_post_gate_unitary_matrix : numpy.ndarray or None, optional
        A 2x2 unitary matrix as pre-post gate operations. Defaults to
        the unitary matrix corresponding to a rotation of :math:'\\pi/2' around
        X-axis. If None, pre-post gate is omitted from the circuit.
    add_measurement : bool, optional
        If True, the circuit contains a measurement operation for each of the
        target qubits. Measurement from each of the qubits is associated
        with a string as key. The string is formatted as 'qubit-X' where
        X is a number between 0 and len(target_qubits).
    circuit_type : str, optional
        One of 'scheduled circuit' or 'standard circuit'. In the case of
        'standard circuit', the circuit will be a sequence of desired operations
        at offsets specified by the supplied dynamic decoupling sequence and the
        duration between any two offsets will have 'identity' gates; the method
        will return a 'cirq.Circuit'. In the case of 'scheduled circuit', the desired
        operations will be scheduled at offsets specified by the dynamic decoupling
        sequence; in this case a 'cirq.Schedule' object is returned.  Both `cirq.Circuit`
        and 'cirq.Schedule' can be used with 'cirq.Simulator'.
        See `Circuits <https://cirq.readthedocs.io/en/stable/circuits.html>` _,
        `Schedules <https://cirq.readthedocs.io/en/stable/schedules.html>` _ and
        `Simulation <https://cirq.readthedocs.io/en/stable/simulation.html>` _.
    device : cirq.Device, optional
        A cirq.Device that specifies hardware constraints for validating circuits
        and schedules. If None, a unconstrained device is used. See `Cirq Documentation
        <https://cirq.readthedocs.io/en/stable/schedules.html/>` _.

    Returns
    -------
    cirq.Circuit or cirq.Schedule
        The circuit or schedule (depending on circuit_type option).
        Either of them can be used with cirq.Simulator.

    Raises
    ------
    ArgumentsValueError
        If any of the input parameters result in an invalid operation.

    Notes
    -----

    Dynamic Decoupling Sequences (DDS) consist of idealized pulse operation. Theoretically,
    these operations (pi-pulses in X,Y or Z) occur instantaneously. However, in practice,
    pulses require time. Therefore, this method of converting an idealized sequence
    results to a circuit that is only an approximate implementation of the idealized sequence.

    In idealized definition of DDS, `offsets` represents the instances within sequence
    `duration` where a pulse occurs instantaneously. A series of appropriate circuit components
    is placed in order to represent these pulses.

    In 'standard circuit', the `gaps` or idle time in between active pulses are filled up
    with `identity` gates. Each identity gate introduces a delay of `gate_time`. In this
    implementation, the number of identity gates is determined by
    :math:`np.int(np.floor(offset_distance / gate_time))`. As a consequence,
    :math:`np.int(np.floor(offset_distance / gate_time))`. As a consequence,
    the duration of the real-circuit is :math:`gate_time \\times number_of_identity_gates +
    pulse_gate_time \\times number_of_pulses`.

    In 'scheduled circuit', the active pulses are scheduled to be activated at a certain
    instant calculated from the start of the sequence. This does not require identity gates
    to be placed between offsets.

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

    if pre_post_gate_unitary_matrix is None:
        pre_post_gate = None
    else:
        pre_post_gate = cirq.SingleQubitMatrixGate(pre_post_gate_unitary_matrix)

    if circuit_type not in [SCHEDULED_CIRCUIT, STANDARD_CIRCUIT]:
        raise ArgumentsValueError('Circuit type must be one of {} or {}'.format(
            SCHEDULED_CIRCUIT, STANDARD_CIRCUIT), {'algorithm': circuit_type})

    if circuit_type == STANDARD_CIRCUIT:
        return _get_standard_circuit(dynamic_decoupling_sequence=dynamic_decoupling_sequence,
                                     target_qubits=target_qubits,
                                     gate_time=gate_time,
                                     pre_post_gate=pre_post_gate,
                                     add_measurement=add_measurement)

    if device is None:
        device = cirq.UnconstrainedDevice

    if not isinstance(device, cirq.Device):
        raise ArgumentsValueError('Device must be a cirq.Device type.',
                                  {'device': device})

    return _get_scheduled_circuit(dynamic_decoupling_sequence=dynamic_decoupling_sequence,
                                  target_qubits=target_qubits,
                                  gate_time=gate_time,
                                  pre_post_gate=pre_post_gate,
                                  add_measurement=add_measurement,
                                  device=device)
