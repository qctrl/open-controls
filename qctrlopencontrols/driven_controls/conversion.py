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
driven_controls.conversion
================
"""

import numpy as np

from qctrlopencontrols.globals import CARTESIAN, CYLINDRICAL
from qctrlopencontrols.exceptions import ArgumentsValueError

def convert_to_standard_segments(transformed_segments, maximum_rabi_rate=None,
                                 coordinates=CARTESIAN, dimensionless=True):
    """Converts the dimensionless segments of any type into dimension-full Cartesian segments

    Parameters
    ----------
    transformed_segments : list
        segments of pulse in [number_of_segments, 4] shape
    maximum_rabi_rate : float
        maximum rabi rate
    coordinates : string
        'cartesian' or 'cylindrical' or 'polar'
        defines the type of the transformed_segments supplied.
        if 'cartesian' - the segments should be in
        [amplitude_x, amplitude_y, amplitude_z,segment_duration] format
        if 'cylindrical' - the segments should be in
        [on_resonance_amplitude, azimuthal_angle, detuning, segment_duration] format
    dimensionless : boolean
        if True, identifies if the transformed_segments are dimensionless

    Returns
    ------
    numpy.ndarray
        Same size as the input segment [number_of_segments, 4]

    The returned array will be equal to the dimension-full cartesian segments of the
    transformed segments

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    # making a copy of the segments otherwise during the call from
    # get_segments(.) the internal segments will be modified
    transformed_segments = np.array(transformed_segments, dtype=np.float)
    segments_copy = np.array(transformed_segments, dtype=np.float)
    number_of_segments = len(segments_copy)
    dimensionless = bool(dimensionless)

    # Raise error if dimensionless is True and maximum_rabi_rate is not a float
    if dimensionless:
        if maximum_rabi_rate is None:
            raise ArgumentsValueError('Maximum rate rate needs to be a valid float',
                                      {'maximum_rabi_rate': maximum_rabi_rate,
                                       'dimensionless': dimensionless},
                                      extras={'segments': transformed_segments,
                                              'number_of_segments': number_of_segments,
                                              'coordinates': coordinates})
        maximum_rabi_rate = float(maximum_rabi_rate)

    # Raise error if segments are not in [number_of_segments, 4] format
    segments_copy = np.array(segments_copy, dtype=np.float)
    if segments_copy.shape != (number_of_segments, 4):
        raise ArgumentsValueError('Segments must be of shape (number_of_segments,4).',
                                  {'segments': transformed_segments},
                                  extras={'number_of_segments': number_of_segments})

    if coordinates == CYLINDRICAL:

        # convert to cartesian
        cos_theta = np.cos(transformed_segments[:, 1])
        sin_theta = np.sin(transformed_segments[:, 1])
        radius = transformed_segments[:, 0]

        segments_copy[:, 0] = radius * cos_theta
        segments_copy[:, 1] = radius * sin_theta

    # if dimensionless, make the segments dimension-full
    if dimensionless:
        segments_copy[:, 0:2] = segments_copy[:, 0:2] * maximum_rabi_rate

    return segments_copy
