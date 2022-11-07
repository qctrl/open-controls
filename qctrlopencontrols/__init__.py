# Copyright 2022 Q-CTRL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Top-level package for Q-CTRL Open Controls.
"""

__version__ = "9.1.4"

from .driven_controls.driven_control import DrivenControl
from .driven_controls.predefined import (
    new_bb1_control,
    new_corpse_control,
    new_corpse_in_bb1_control,
    new_corpse_in_scrofulous_control,
    new_corpse_in_sk1_control,
    new_drag_control,
    new_gaussian_control,
    new_modulated_gaussian_control,
    new_primitive_control,
    new_scrofulous_control,
    new_sk1_control,
    new_wamf1_control,
)
from .dynamic_decoupling_sequences.dynamic_decoupling_sequence import (
    DynamicDecouplingSequence,
    convert_dds_to_driven_control,
)
from .dynamic_decoupling_sequences.predefined import (
    new_carr_purcell_sequence,
    new_cpmg_sequence,
    new_periodic_sequence,
    new_quadratic_sequence,
    new_ramsey_sequence,
    new_spin_echo_sequence,
    new_uhrig_sequence,
    new_walsh_sequence,
    new_x_concatenated_sequence,
    new_xy_concatenated_sequence,
)

__all__ = [
    "DrivenControl",
    "DynamicDecouplingSequence",
    "convert_dds_to_driven_control",
    "new_bb1_control",
    "new_corpse_control",
    "new_corpse_in_bb1_control",
    "new_corpse_in_scrofulous_control",
    "new_corpse_in_sk1_control",
    "new_gaussian_control",
    "new_modulated_gaussian_control",
    "new_drag_control",
    "new_primitive_control",
    "new_scrofulous_control",
    "new_sk1_control",
    "new_wamf1_control",
    "new_carr_purcell_sequence",
    "new_cpmg_sequence",
    "new_periodic_sequence",
    "new_quadratic_sequence",
    "new_ramsey_sequence",
    "new_spin_echo_sequence",
    "new_uhrig_sequence",
    "new_walsh_sequence",
    "new_x_concatenated_sequence",
    "new_xy_concatenated_sequence",
]
