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
Tests converstion to Qiskit Circuit
===================================
"""

import numpy as np

from qiskit import execute
from qiskit import BasicAer

from qctrlopencontrols import (
    new_predefined_dds, convert_dds_to_quantum_circuit)


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

        dd_sequence_params['number_outer_offsets'] = 4
        dd_sequence_params['number_inner_offsets'] = 4

    elif dd_sequence_params['scheme'] in ['X concatenated',
                                          'XY concatenated']:

        dd_sequence_params['concatenation_order'] = 2

    sequence = new_predefined_dds(**dd_sequence_params)
    return sequence


def _check_circuit_unitary(pre_post_gate_parameters, multiplier):
    """Check the unitary of a dynamic decoupling operation
    """

    backend = 'unitary_simulator'
    number_of_shots = 1
    backend_simulator = BasicAer.get_backend(backend)

    for sequence_scheme in ['Carr-Purcell', 'Carr-Purcell-Meiboom-Gill',
                            'Uhrig single-axis', 'periodic single-axis', 'Walsh single-axis',
                            'quadratic', 'X concatenated',
                            'XY concatenated']:
        sequence = _create_test_sequence(sequence_scheme)
        quantum_circuit = convert_dds_to_quantum_circuit(
            dynamic_decoupling_sequence=sequence,
            pre_post_gate_parameters=pre_post_gate_parameters,
            add_measurement=False, algorithm='instant unitary')

        job = execute(quantum_circuit,
                      backend_simulator,
                      shots=number_of_shots)
        result = job.result()
        unitary = result.get_unitary(quantum_circuit)

        assert np.allclose(np.array([[1, 0], [0, 1]]),
                           np.abs(
                               np.dot(np.linalg.inv(multiplier),
                                      np.dot(unitary, np.linalg.inv(multiplier)))))


def test_identity_operation():

    """Tests if the Dynamic Decoupling Sequence gives rise to Identity
    operation in Qiskit
    """
    _multiplier = np.array([[1, 0], [0, 1]])
    _check_circuit_unitary([0., 0., 0.], _multiplier)
    _check_circuit_unitary(None, _multiplier)

    _multiplier = (1. / np.power(2, 0.5)) * np.array([[1, -1j], [-1j, 1]], dtype='complex')
    _check_circuit_unitary([np.pi / 2, -np.pi / 2, np.pi / 2], _multiplier)

if __name__ == '__main__':
    pass
