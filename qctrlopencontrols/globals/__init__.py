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
