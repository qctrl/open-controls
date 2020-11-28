# Copyright 2020 Q-CTRL Pty Ltd & Q-CTRL Inc
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
Defines commonly used constants.
"""

import numpy as np

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex)
SIGMA_M = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex)
SIGMA_P = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex)


# Defines constants for driven controls module.

# Maximum allowed rabi rate
UPPER_BOUND_RABI_RATE = 1e10

# Maximum allowed detuning rate
UPPER_BOUND_DETUNING_RATE = UPPER_BOUND_RABI_RATE

# Maximum allowed duration of a control
UPPER_BOUND_DURATION = 1e6

# Minimum allowed duration of a control
LOWER_BOUND_DURATION = 1e-12

# Maximum number of segments allowed in a control
UPPER_BOUND_SEGMENTS = 10000

# Defines constants for dynamical decoupling module.

# Maximum number of offsets allowed in a Dynamical Decoupling sequence.
UPPER_BOUND_OFFSETS = 10000

# Matplotlib format of data for plotting
MATPLOTLIB = "matplotlib"
