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
Tests converstion to Pyquil program
===================================
"""

import numpy as np

from pyquil.gates import RX, RY, RZ, I
from pyquil.quil import Pragma

from qctrlopencontrols import (
    DynamicDecouplingSequence,
    convert_dds_to_pyquil_program)

def test_pyquil_program():

    """Tests if the Dynamic Decoupling Sequence gives rise to Identity
    operation in Pyquil
    """
    _duration = 5e-6
    _offsets = [0, 1e-6, 2.5e-6, 4e-6, 5e-6]
    _rabi_rotations = [np.pi / 2, np.pi / 2, np.pi, 0, np.pi / 2]
    _azimuthal_angles = [0, 0, np.pi / 2, 0, 0]
    _detuning_rotations = [0, 0, 0, np.pi, 0]

    sequence = DynamicDecouplingSequence(
        duration=_duration,
        offsets=_offsets,
        rabi_rotations=_rabi_rotations,
        azimuthal_angles=_azimuthal_angles,
        detuning_rotations=_detuning_rotations)

    program = convert_dds_to_pyquil_program(
        sequence,
        [0],
        gate_time=1e-6)

    assert len(program) == 13
    assert program[0] == Pragma("PRESERVE_BLOCK")
    assert program[-1] == Pragma("END_PRESERVE_BLOCK")
    assert program[1] == RX(np.pi/2, 0)
    assert program[2] == I(0)
    assert program[3] == RX(np.pi / 2, 0)
    assert program[4] == I(0)
    assert program[5] == RY(np.pi, 0)
    assert program[6] == I(0)
    assert program[7] == RZ(np.pi, 0)
    assert program[8] == I(0)
    assert program[9] == RX(np.pi / 2, 0)

if __name__ == '__main__':
    test_pyquil_program()
