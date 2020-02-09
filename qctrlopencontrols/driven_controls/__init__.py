# Copyright 2020 Q-CTRL Pty Ltd & Q-CTRL Inc. All rights reserved.
#
# Licensed under the Q-CTRL Terms of service (the "License"). Unauthorized
# copying or use of this file, via any medium, is strictly prohibited.
# Proprietary and confidential. You may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://q-ctrl.com/terms
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS. See the
# License for the specific language.

"""
======================
driven_controls module
======================
"""
##### Maximum and Minimum bounds ######

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

##### Types of driven controls ######

PRIMITIVE = 'primitive'
"""Primitive control
"""

BB1 = 'BB1'
"""First-order Wimperis control, also known as BB1
"""

SK1 = 'SK1'
"""First-order Solovay-Kitaev control
"""

WAMF1 = 'WAMF1'
"""First-order Walsh sequence control
"""

CORPSE = 'CORPSE'
"""Dynamically corrected control - Compensating for Off-Resonance with a Pulse Sequence (COPRSE)
"""

CORPSE_IN_BB1 = 'CORPSE in BB1'
"""Concatenated dynamically corrected control - BB1 inside COPRSE
"""

CORPSE_IN_SK1 = 'CORPSE in SK1'
"""Concatenated dynamically corrected control - First order Solovay-Kitaev inside COPRSE
"""

SCROFULOUS = 'SCROFULOUS'
"""Dynamically corrected control -
   Short Composite Rotation For Undoing Length Over and Under Shoot (SCROFULOUS)
"""

CORPSE_IN_SCROFULOUS = 'CORPSE in SCROFULOUS'
"""Concatenated dynamically corrected control - CORPSE inside SCROFULOUS
"""
