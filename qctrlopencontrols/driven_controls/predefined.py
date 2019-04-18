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
===================
driven_controls.predefined
===================
"""

import numpy as np

from qctrlopencontrols.exceptions import ArgumentsValueError
from qctrlopencontrols import DrivenControls

from .constants import (
    UPPER_BOUND_RABI_RATE, UPPER_BOUND_DETUNING_RATE,
    UPPER_BOUND_DURATION, LOWER_BOUND_DURATION, UPPER_BOUND_SEGMENTS,
    PRIMITIVE, WIMPERIS_1, SOLOVAY_KITAEV_1,
    WALSH_AMPLITUDE_MODULATED_FILTER_1,
    COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE,
    COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_SOLOVAY_KITAEV,
    COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_WIMPERIS,
    SHORT_COMPOSITE_ROTATION_FOR_UNDOING_LENGTH_OVER_AND_UNDER_SHOOT,
    CORPSE_IN_SCROFULOUS_PULSE)
