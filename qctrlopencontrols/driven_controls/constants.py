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
driven_controls.constants
================
"""

GAUSSIAN_STANDARD_DEVIATION_SCALE = 6.
"""
"""

#maximum and minimum values
UPPER_BOUND_RABI_RATE = 1e10
"""Maximum allowed rabi rate
"""

UPPER_BOUND_DETUNING_RATE = UPPER_BOUND_RABI_RATE
"""Maximum allowed detuning rate
"""

UPPER_BOUND_DURATION = 1e6
"""Maximum allowed duration of a control
"""

LOWER_BOUND_DURATION = 1e-12
"""Minimum allowed duration of a control
"""

UPPER_BOUND_SEGMENTS = 10000
"""Maximum number of segments allowed in a control
"""

#Driven control types
PRIMITIVE = 'primitive'
"""Primitive control
"""

WIMPERIS_1 = 'wimperis_1'
"""First-order Wimperis control, also known as BB1
"""

SOLOVAY_KITAEV_1 = 'solovay_kitaev_1'
"""First-order Solovay-Kitaev control
"""

WALSH_AMPLITUDE_MODULATED_FILTER_1 = 'walsh_amplitude_modulated_filter_1'
"""First-order Walsh sequence control
"""

COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE = \
    'compensating_for_off_resonance_with_a_pulse_sequence'
"""Dynamically corrected control - commonly abbreviated as COPRSE
"""

COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_WIMPERIS = \
    'compensating_for_off_resonance_with_a_pulse_sequence_with_wimperis'
"""Concatenated dynamically corrected control - Wimperis inside COPRSE
"""

COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_SOLOVAY_KITAEV = \
    'compensating_for_off_resonance_with_a_pulse_sequence_with_solovay_kitaev'
"""Concatenated dynamically corrected control - Solovay-Kitaev inside COPRSE
"""

SHORT_COMPOSITE_ROTATION_FOR_UNDOING_LENGTH_OVER_AND_UNDER_SHOOT = \
    'short_composite_rotation_for_undoing_length_over_and_under_shoot'
"""Dynamically corrected control - commonly abbreviated as SCROFULOUS
"""

CORPSE_IN_SCROFULOUS_PULSE = 'corpse_in_scrofulous_pulse'
"""Concatenated dynamically corrected control - CORPSE inside SCROFULOUS
"""
