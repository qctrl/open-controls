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
Top-level package for Q-CTRL Open Controls.
"""

__version__ = "4.2.1"

from .driven_controls.driven_control import DrivenControl
from .driven_controls.predefined import new_predefined_driven_control

from .dynamic_decoupling_sequences.dynamic_decoupling_sequence import DynamicDecouplingSequence
from .dynamic_decoupling_sequences.predefined import new_predefined_dds
from .dynamic_decoupling_sequences.driven_controls import convert_dds_to_driven_control

__all__ = ['convert_dds_to_driven_control',
           'new_predefined_dds',
           'new_predefined_driven_control',
           'DrivenControl',
           'DynamicDecouplingSequence']
