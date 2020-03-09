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
============================
dynamic_decoupling_sequences
============================
"""

UPPER_BOUND_OFFSETS = 10000
"""Maximum number of offsets allowed in a Dynamical
Decoupling sequence.
"""

MATPLOTLIB = 'matplotlib'
"""Matplotlib format of data for plotting
"""

###### Types of Dynamic Decoupling Sequences #######

RAMSEY = 'Ramsey'
"""Ramsey sequence
"""

SPIN_ECHO = 'spin echo'
"""Spin echo (SE) dynamical decoupling sequence
"""

CARR_PURCELL = 'Carr-Purcell'
"""Carr-Purcell (CP) dynamical decoupling sequence
"""

CARR_PURCELL_MEIBOOM_GILL = 'Carr-Purcell-Meiboom-Gill'
"""Carr-Purcell-Meiboom-Gill (CPMG) dynamical decoupling sequence
"""

UHRIG_SINGLE_AXIS = 'Uhrig single-axis'
"""Uhrig (single-axis) dynamical decoupling sequence
"""

PERIODIC_SINGLE_AXIS = 'periodic single-axis'
"""Periodical dynamical decoupling sequence
"""

WALSH_SINGLE_AXIS = 'Walsh single-axis'
"""Walsh dynamical decoupling sequence
"""

QUADRATIC = 'quadratic'
"""Quadratic dynamical decoupling sequence
"""

X_CONCATENATED = 'X concatenated'
"""X-Concatenated dynamical decoupling sequence
"""

XY_CONCATENATED = 'XY concatenated'
"""XY-Concatenated dynamical decoupling sequence
"""
