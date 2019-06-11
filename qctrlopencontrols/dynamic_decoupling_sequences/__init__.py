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
================
Sequences module
================
"""

from .constants import (
    UPPER_BOUND_OFFSETS, SPIN_ECHO, CARR_PURCELL,
    CARR_PURCELL_MEIBOOM_GILL, UHRIG_SINGLE_AXIS,
    PERIODIC_SINGLE_AXIS, WALSH_SINGLE_AXIS, QUADRATIC,
    X_CONCATENATED, XY_CONCATENATED)

from .dynamic_decoupling_sequence import DynamicDecouplingSequence
from .predefined import new_predefined_dds
from .driven_controls import convert_dds_to_driven_control

__all__ = ['CARR_PURCELL',
           'CARR_PURCELL_MEIBOOM_GILL',
           'UPPER_BOUND_OFFSETS',
           'PERIODIC_SINGLE_AXIS',
           'QUADRATIC',
           'SPIN_ECHO',
           'UHRIG_SINGLE_AXIS',
           'WALSH_SINGLE_AXIS',
           'X_CONCATENATED',
           'XY_CONCATENATED',
           'DynamicDecouplingSequence',
           'convert_dds_to_driven_control',
           'new_predefined_dds']
