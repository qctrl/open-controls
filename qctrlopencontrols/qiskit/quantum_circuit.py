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
======================
qiskit.quantum_circuit
======================
"""

import numpy as np

from qiskit import (
    QuantumRegister, ClassicalRegister, QuantumCircuit)
from qiskit.qasm import pi

from qctrlopencontrols.dynamic_decoupling_sequences import DynamicDecouplingSequence
from qctrlopencontrols.exceptions import ArgumentsValueError

from .constants import (FIX_DURATION_UNITARY, INSTANT_UNITARY)


def _get_circuit_gate_list(dynamic_decoupling_sequence,
                           gate_time,
                           unitary_time):

    """Converts the operations in a sequence into list of gates
    of a quantum circuit

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

    """Returns the pulses based on the rotation operation

    Parameters
    ----------
    operation : numpy.ndarray
        1-D array (length=3) consisting of rabi rotation, azimuthal_angle
        and detuning_rotation at an offset of a sequence

    Returns
    -------
    numpy.ndarray
        A 1-D array of length 3 containing x_rotation, y_rotation and z_rotation
        calculated from sequence operation
    """

    x_rotation = operation[0] * np.cos(operation[1])
    y_rotation = operation[0] * np.sin(operation[1])
    z_rotation = operation[2]

    pulses = np.array([x_rotation, y_rotation, z_rotation])

    return pulses


def convert_dds_to_quantum_circuit(
        dynamic_decoupling_sequence,
        target_qubits=None,
        gate_time=0.1,
        add_measurement=True,
        algorithm=INSTANT_UNITARY,
        quantum_registers=None,
        circuit_name=None):

    """Converts a Dynamic Decoupling Sequence into QuantumCircuit
    as defined in Qiskit
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
    quantum_registers : QuantumRegister, optional
        The set of quantum registers; defaults to None
        If not None, it must have the target qubit specified in `target_qubit`
        indices list
    circuit_name : str, optional
        A string indicating the name of the circuit; defaults to None

    Returns
    -------
    QuantumCircuit
        The circuit defined from the specified dynamic decoupling sequence

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
    `duration` where a pulse occurs instantaneously. A series of appropriate circuit component
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

    if quantum_registers is not None:
        if (max(target_qubits)+1) > len(quantum_registers):
            raise ArgumentsValueError('Target qubit is not present in quantum_registers',
                                      {'target_qubits': target_qubits,
                                       'size(quantum_registers)': len(quantum_registers)},
                                      extras={'max(target_qubits)': max(target_qubits)})
        quantum_registers = quantum_registers
    else:
        quantum_registers = QuantumRegister(max(target_qubits)+1)

    classical_registers = None
    if add_measurement:
        classical_registers = ClassicalRegister(len(target_qubits))
        quantum_circuit = QuantumCircuit(quantum_registers, classical_registers)
    else:
        quantum_circuit = QuantumCircuit(quantum_registers)

    if circuit_name is not None:
        quantum_circuit.name = circuit_name

    unitary_time = 0.
    if algorithm == FIX_DURATION_UNITARY:
        unitary_time = gate_time

    circuit_gate_list = _get_circuit_gate_list(
        dynamic_decoupling_sequence=dynamic_decoupling_sequence,
        gate_time=gate_time,
        unitary_time=unitary_time)

    offset_count = 0
    for gate in circuit_gate_list:

        if gate == 'id':
            for qubit in target_qubits:
                quantum_circuit.iden(quantum_registers[qubit])  # pylint: disable=no-member
                quantum_circuit.barrier(quantum_registers[qubit])  # pylint: disable=no-member
            continue

        instance_operation = np.array([dynamic_decoupling_sequence.rabi_rotations[offset_count],
                                       dynamic_decoupling_sequence.azimuthal_angles[offset_count],
                                       dynamic_decoupling_sequence.detuning_rotations[offset_count]
                                       ])

        rotations = _get_rotations(instance_operation)
        nonzero_pulse_counts = 0
        for rotation in rotations:
            if not np.isclose(rotation, 0.0):
                nonzero_pulse_counts += 1
        if nonzero_pulse_counts > 1:
            raise ArgumentsValueError('Open Controls support a sequence with one '
                                      'valid pulse at any offset. Found sequence '
                                      'with multiple rotation operations at an offset.',
                                      {'dynamic_decoupling_sequence': str(
                                          dynamic_decoupling_sequence),
                                       'instance_operation': instance_operation})
        for qubit in target_qubits:
            if nonzero_pulse_counts == 0:
                quantum_circuit.u3(0., 0., 0.,  #pylint: disable=no-member
                                   quantum_registers[qubit])
            else:
                if not np.isclose(rotations[0], 0.0):
                    quantum_circuit.u3(rotations[0], -pi/2, pi/2, #pylint: disable=no-member
                                       quantum_registers[qubit])
                elif not np.isclose(rotations[1], 0.0):
                    quantum_circuit.u3(rotations[1], 0., 0.,  #pylint: disable=no-member
                                       quantum_registers[qubit])
                elif not np.isclose(rotations[2], 0.):
                    quantum_circuit.u1(rotations[2],      #pylint: disable=no-member
                                       quantum_registers[qubit])
            quantum_circuit.barrier(quantum_registers[qubit])    #pylint: disable=no-member

        offset_count += 1

    if add_measurement:
        for q_index, qubit in enumerate(target_qubits):
            quantum_circuit.measure(quantum_registers[qubit],   #pylint: disable=no-member
                                    classical_registers[q_index])

    return quantum_circuit
