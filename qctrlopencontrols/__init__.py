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
qctrlopencontrols
=================
"""

__version__ = "2.0.1"

from .cirq.circuit import convert_dds_to_cirq_circuit
from .cirq.schedule import convert_dds_to_cirq_schedule

from .driven_controls.driven_control import DrivenControl
from .driven_controls.predefined import new_predefined_driven_control

from .dynamic_decoupling_sequences.dynamic_decoupling_sequence import DynamicDecouplingSequence
from .dynamic_decoupling_sequences.predefined import new_predefined_dds
from .dynamic_decoupling_sequences.driven_controls import convert_dds_to_driven_control

from .pyquil.program import convert_dds_to_pyquil_program

from .qiskit.quantum_circuit import convert_dds_to_qiskit_quantum_circuit

__all__ = ['convert_dds_to_cirq_circuit',
           'convert_dds_to_cirq_schedule',
           'convert_dds_to_driven_control',
           'convert_dds_to_pyquil_program',
           'convert_dds_to_qiskit_quantum_circuit',
           'new_predefined_dds',
           'new_predefined_driven_control',
           'DrivenControl',
           'DynamicDecouplingSequence']
