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
==============
globals module
==============
"""

QCTRL_EXPANDED = 'Q-CTRL expanded'
"""Defines the export file format to be in Q-CTRL
Expanded format
"""

CSV = 'CSV'
"""Defines the CSV file type for control export
"""

JSON = 'JSON'
"""Defines the JSON file type for control export
"""

#coordinate system labels
CARTESIAN = 'cartesian'
"""Defines Cartesian coordinate system
"""

CYLINDRICAL = 'cylindrical'
"""Defines Cylindrical coordinate system
"""

FIX_DURATION_UNITARY = 'fixed duration unitary'
"""Algorithm to convert a DDS to Quantum circuit
where the unitaries are considered as gates with finite duration
"""

INSTANT_UNITARY = 'instant unitary'
"""Algorithm to convert a DDS to Quantum circuit where the
unitaties are considered as instantaneous operation.
"""
