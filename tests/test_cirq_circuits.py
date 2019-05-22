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
===================================
Tests converstion to Cirq Circuit
===================================
"""

import numpy as np
import cirq

from qctrlopencontrols import (
    new_predefined_dds, convert_dds_to_cirq_circuit)


def _create_test_sequence(sequence_scheme):

    """Create a DD sequence of choice'''

    Parameters
    ----------
    sequence_scheme : str
        One of 'Spin echo', 'Carr-Purcell', 'Carr-Purcell-Meiboom-Gill',
        'Uhrig single-axis', 'Periodic single-axis', 'Walsh single-axis',
        'Quadratic', 'X concatenated',
        'XY concatenated'

    Returns
    -------
    DynamicDecouplingSequence
        The Dynamical Decoupling Sequence instance built from supplied
        schema information
    """

    dd_sequence_params = dict()
    dd_sequence_params['scheme'] = sequence_scheme
    dd_sequence_params['duration'] = 4

    # 'spin_echo' does not need any additional parameter

    if dd_sequence_params['scheme'] in ['Carr-Purcell', 'Carr-Purcell-Meiboom-Gill',
                                        'Uhrig single-axis', 'periodic single-axis']:

        dd_sequence_params['number_of_offsets'] = 2

    elif dd_sequence_params['scheme'] in ['Walsh single-axis']:

        dd_sequence_params['paley_order'] = 5

    elif dd_sequence_params['scheme'] in ['quadratic']:

        dd_sequence_params['duration'] = 16
        dd_sequence_params['number_outer_offsets'] = 4
        dd_sequence_params['number_inner_offsets'] = 4

    elif dd_sequence_params['scheme'] in ['X concatenated',
                                          'XY concatenated']:

        dd_sequence_params['duration'] = 16
        dd_sequence_params['concatenation_order'] = 2

    sequence = new_predefined_dds(**dd_sequence_params)
    return sequence


def _check_circuit_output(pre_post_gate_unitary_matrix,
                          circuit_type, expected_result):
    """Check the unitary of a dynamic decoupling operation
    """

    simulator = cirq.Simulator()
    for sequence_scheme in ['Carr-Purcell', 'Carr-Purcell-Meiboom-Gill',
                            'Uhrig single-axis', 'periodic single-axis',
                            'Walsh single-axis', 'quadratic', 'X concatenated',
                            'XY concatenated']:
        sequence = _create_test_sequence(sequence_scheme)
        cirq_circuit = convert_dds_to_cirq_circuit(
            dynamic_decoupling_sequence=sequence,
            pre_post_gate_unitary_matrix=pre_post_gate_unitary_matrix,
            add_measurement=True, circuit_type=circuit_type)

        results = simulator.run(cirq_circuit)
        assert results.measurements['qubit-0'] == expected_result


def test_identity_operation():

    """Tests if the Dynamic Decoupling Sequence gives rise to expected
    state with different pre-post gates
    """
    _check_circuit_output(None, 'scheduled circuit', 0)
    pre_post_gate_unitary_matrix = (1. / np.power(2, 0.5)) * np.array(
        [[1, -1j], [-1j, 1]], dtype='complex')
    _check_circuit_output(pre_post_gate_unitary_matrix,
                          'scheduled circuit', 1)

    pre_post_gate_unitary_matrix = np.array(
        [[1, 0], [0, 1]], dtype='complex')
    _check_circuit_output(pre_post_gate_unitary_matrix,
                          'scheduled circuit', 0)

    _check_circuit_output(None, 'standard circuit', 0)
    pre_post_gate_unitary_matrix = (1. / np.power(2, 0.5)) * np.array(
        [[1, -1j], [-1j, 1]], dtype='complex')
    _check_circuit_output(pre_post_gate_unitary_matrix,
                          'standard circuit', 1)

    pre_post_gate_unitary_matrix = np.array(
        [[1, 0], [0, 1]], dtype='complex')
    _check_circuit_output(pre_post_gate_unitary_matrix,
                          'standard circuit', 0)

if __name__ == '__main__':
    pass
