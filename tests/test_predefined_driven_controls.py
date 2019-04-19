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
====================================
Tests for Predefined Driven Controls
====================================
"""

import os
os.chdir('/home/virginia/Documents/qctrl/python-open-controls')  

import numpy as np
import pytest

from qctrlopencontrols.exceptions import ArgumentsValueError

from qctrlopencontrols.driven_controls import (
    new_primitive_control, new_wimperis_1_control, new_solovay_kitaev_1_control,
    new_compensating_for_off_resonance_with_a_pulse_sequence_control,
    new_compensating_for_off_resonance_with_a_pulse_sequence_with_solovay_kitaev_control,
    new_compensating_for_off_resonance_with_a_pulse_sequence_with_wimperis_control,
    new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control,
    new_walsh_amplitude_modulated_filter_1_control,
    new_corpse_in_scrofulous_control
)

from qctrlopencontrols.globals import SQUARE


def test_primitive_control_segments():
    """Test the segments predefined primitive driven control
    """
    _rabi_rate = 1
    _rabi_rotation = np.pi
    _azimuthal_angle = np.pi/2
    _segments = [[
        _rabi_rate * np.cos(_azimuthal_angle),
        _rabi_rate * np.sin(_azimuthal_angle),
        0.,
        _rabi_rotation / _rabi_rate], ]

    primitive_control = new_primitive_control(
        rabi_rotation=_rabi_rotation,
        maximum_rabi_rate=_rabi_rate,
        azimuthal_angle=_azimuthal_angle,
        shape=SQUARE
    )

    assert np.allclose(_segments, primitive_control.segments)
