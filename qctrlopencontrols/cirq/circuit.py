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

from qctrlopencontrols.qiskit import (
    FIX_DURATION_UNITARY, INSTANT_UNITARY,
    get_circuit_gate_list, get_rotations)


def _get_standard_circuit(dynamic_decoupling_sequence,
                          target_qubits,
                          gate_time,
                          algorithm,
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
    algorithm : str, optional
        One of 'fixed duration unitary' or 'instant unitary'; In the case of
        'fixed duration unitary', the operations are assumed to be taking the amount of
        gate_time while 'instant unitary' assumes unitaries to be instantaneous;
        defaults to 'instant unitary'. Note that this option is only used for
        'standard circuit'; 'scheduled circuit' always contains a 'fixed duration unitary'.
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

    unitary_time = 0.
    if algorithm == FIX_DURATION_UNITARY:
        unitary_time = gate_time

    circuit_gate_list = get_circuit_gate_list(
        dynamic_decoupling_sequence=dynamic_decoupling_sequence,
        gate_time=gate_time,
        unitary_time=unitary_time)

    circuit = cirq.Circuit()

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
        add_measurement=True,
        algorithm=INSTANT_UNITARY):

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
    add_measurement : bool, optional
        If True, the circuit contains a measurement operation for each of the
        target qubits. Measurement from each of the qubits is associated
        with a string as key. The string is formatted as 'qubit-X' where
        X is a number between 0 and len(target_qubits).
    algorithm : str, optional
        One of 'fixed duration unitary' or 'instant unitary'; In the case of
        'fixed duration unitary', the sequence operations are assumed to be
        taking the amount of gate_time while 'instant unitary' assumes the sequence
        operations are instantaneous (and hence does not contribute to the delay between
        offsets). Defaults to 'instant unitary'. Note that this option is only used for
        'standard circuit'; 'scheduled circuit' always contains a 'fixed duration unitary'.

    Returns
    -------
    cirq.Circuit
        The circuit containing gates corresponding to sequence operation.

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

    if algorithm not in [FIX_DURATION_UNITARY, INSTANT_UNITARY]:
        raise ArgumentsValueError('Algorithm must be one of {} or {}'.format(
            INSTANT_UNITARY, FIX_DURATION_UNITARY), {'algorithm': algorithm})

    return _get_standard_circuit(dynamic_decoupling_sequence=dynamic_decoupling_sequence,
                                 target_qubits=target_qubits,
                                 gate_time=gate_time,
                                 algorithm=algorithm,
                                 add_measurement=add_measurement)
