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
======================
driven_controls module
======================
"""

from .driven_control import DrivenControl

from .constants import (
    UPPER_BOUND_RABI_RATE, UPPER_BOUND_DETUNING_RATE,
    UPPER_BOUND_DURATION, LOWER_BOUND_DURATION, UPPER_BOUND_SEGMENTS,
    PRIMITIVE, BB1, SK1,
    WAMF1,
    CORPSE,
    CORPSE_IN_SK1,
    CORPSE_IN_BB1,
    SCROFULOUS,
    CORPSE_IN_SCROFULOUS)

from .predefined import new_predefined_driven_control

__all__ = ['UPPER_BOUND_RABI_RATE', 'UPPER_BOUND_DETUNING_RATE',
           'UPPER_BOUND_DURATION', 'LOWER_BOUND_DURATION', 'UPPER_BOUND_SEGMENTS',
           'PRIMITIVE', 'BB1', 'SK1',
           'WAMF1', 'CORPSE', 'CORPSE_IN_SK1', 'CORPSE_IN_BB1', 'SCROFULOUS',
           'CORPSE_IN_SCROFULOUS', 'new_predefined_driven_control',
           'DrivenControl']
