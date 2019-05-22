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
=====================
cirq.constants module
=====================
"""

import numpy as np

SCHEDULED_CIRCUIT = 'scheduled circuit'
"""Constructs the circuit as schedule of rotation
operations at specified offsets. Scheduled circuit
only contains gates specific to desired rotation operations.
"""

STANDARD_CIRCUIT = 'standard circuit'
"""Constructs the circuit as a series of operations that include
identity gates between desired rotation operations.
"""

DEFAULT_ROTATION_MATRIX = (1. / np.power(2, 0.5)) * np.array(
    [[1, -1j], [-1j, 1]], dtype='complex')
"""Unitary matrix for a :math:`\\pi/2` rotation around X-axis.
"""
