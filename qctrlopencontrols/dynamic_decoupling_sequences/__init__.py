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
Defines constants for dynamical decoupling module.
"""

# Maximum number of offsets allowed in a Dynamical Decoupling sequence.
UPPER_BOUND_OFFSETS = 10000

# Matplotlib format of data for plotting
MATPLOTLIB = "matplotlib"


# Ramsey sequence
RAMSEY = "Ramsey"

# Spin echo (SE) dynamical decoupling sequence
SPIN_ECHO = "spin echo"

# Carr-Purcell (CP) dynamical decoupling sequence
CARR_PURCELL = "Carr-Purcell"

# Carr-Purcell-Meiboom-Gill (CPMG) dynamical decoupling sequence
CARR_PURCELL_MEIBOOM_GILL = "Carr-Purcell-Meiboom-Gill"

# Uhrig (single-axis) dynamical decoupling sequence
UHRIG_SINGLE_AXIS = "Uhrig single-axis"

# Periodical dynamical decoupling sequence
PERIODIC_SINGLE_AXIS = "periodic single-axis"

# Walsh dynamical decoupling sequence
WALSH_SINGLE_AXIS = "Walsh single-axis"

# Quadratic dynamical decoupling sequence
QUADRATIC = "quadratic"

# X-Concatenated dynamical decoupling sequence
X_CONCATENATED = "X concatenated"

# XY-Concatenated dynamical decoupling sequence
XY_CONCATENATED = "XY concatenated"
