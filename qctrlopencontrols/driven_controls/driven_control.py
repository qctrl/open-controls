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
driven_controls.driven_controls
======================
"""
import json
import numpy as np

from qctrlopencontrols.exceptions import ArgumentsValueError
from qctrlopencontrols.base import QctrlObject

from qctrlopencontrols.globals import (
    QCTRL_EXPANDED, CSV, JSON, CARTESIAN, CYLINDRICAL)

from .constants import (
    UPPER_BOUND_SEGMENTS, UPPER_BOUND_RABI_RATE, UPPER_BOUND_DETUNING_RATE,
    UPPER_BOUND_DURATION, LOWER_BOUND_DURATION)

def get_plot_data_from_segments(segments):
    """
    Generates arrays that can be used to produce a plot representing the shape of the driven control
    constructed from the segments.

    Parameters
    ----------
    segments : list
        List of segments formatted as described in qctrlopencontrols.driven_controls.DrivenControl

    Returns
    -------
    tuple
        Tuple made up of arrays for plotting formatted as
        (amplitude_x,amplitude_y,amplitude_z,time) where:
        - amplitude_k is the amplitude values.
        - times the time corresponding to each amplitude_k coordinate.
        Note that plot will have repeated times and for amplitudes, this is because it is
        expected that these coordinates are to be used with plotting software that 'joins
        the dots' with linear lines between each coordinate. The time array gives the x
        values for all the amplitude arrays, which give the y values.

    """
    segment_times = np.insert(np.cumsum(segments[:, 3]), 0, 0.)
    coords = len(segment_times)
    coord_amplitude_x = np.concatenate([[0.], segments[:, 0], [0.]])
    coord_amplitude_y = np.concatenate([[0.], segments[:, 1], [0.]])
    coord_amplitude_z = np.concatenate([[0.], segments[:, 2], [0.]])
    plot_time = []
    plot_amplitude_x = []
    plot_amplitude_y = []
    plot_amplitude_z = []
    for i in range(coords):
        plot_time.append(segment_times[i])
        plot_time.append(segment_times[i])
        plot_amplitude_x.append(coord_amplitude_x[i])
        plot_amplitude_x.append(coord_amplitude_x[i + 1])
        plot_amplitude_y.append(coord_amplitude_y[i])
        plot_amplitude_y.append(coord_amplitude_y[i + 1])
        plot_amplitude_z.append(coord_amplitude_z[i])
        plot_amplitude_z.append(coord_amplitude_z[i + 1])

    return (np.array(plot_amplitude_x),
            np.array(plot_amplitude_y),
            np.array(plot_amplitude_z),
            np.array(plot_time))



class DrivenControl(QctrlObject):   #pylint: disable=too-few-public-methods
    """
    Creates a driven control. A driven is a set of segments made up of amplitude vectors
    and corresponding durations.

    Parameters
    ----------
    segments : list, optional
        Defaults to None. A list of amplitude vector components
        and durations. Each element of the list should be formatted as
        [amplitude_x,amplitude_y,amplitude_z,duration] where amplitude_i is the angular
        rabi frequency to be multiplied by the
        corresponding pauli matrix, i.e. amplitude_x would correspond to sigma_x.
        The duration is the time of that segment.
        If None, defaults to a square pi pulse [[np.pi, 0, 0, 1], ].
    name : string, optional
        Defaults to None. An optional string to name the driven control.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    def __init__(self,
                 segments=None,
                 name=None):

        self.name = name
        if self.name is not None:
            self.name = str(self.name)

        if segments is None:
            segments = [[np.pi, 0, 0, 1], ]

        self.segments = np.array(segments, dtype=np.float)
        self.number_of_segments = len(self.segments)
        if self.segments.shape != (self.number_of_segments, 4):
            raise ArgumentsValueError('Segments must be of shape (number_of_segments,4).',
                                      {'segments': self.segments},
                                      extras={'number_of_segments': self.number_of_segments})
        if self.number_of_segments > UPPER_BOUND_SEGMENTS:
            raise ArgumentsValueError(
                'The number of segments must be smaller than the upper bound:'
                + str(UPPER_BOUND_SEGMENTS),
                {'segments': self.segments},
                extras={'number_of_segments': self.number_of_segments})

        self.amplitudes = np.sqrt(np.sum(self.segments[:, 0:3] ** 2, axis=1))

        self.segment_durations = self.segments[:, 3]
        if np.any(self.segment_durations <= 0):
            raise ArgumentsValueError('Duration of driven control segments must all be greater'
                                      + ' than zero.',
                                      {'segments': self.segments},
                                      extras={'segment_durations': self.segment_durations})

        super(DrivenControl, self).__init__(
            base_attributes=['segments', 'shape', 'scheme', 'name'])

        self.angles = self.amplitudes * self.segment_durations
        self.directions = np.array([self.segments[i, 0:3] / self.amplitudes[i]
                                    if self.amplitudes[i] != 0. else np.zeros([3, ])
                                    for i in range(self.number_of_segments)])

        self.segment_times = np.insert(
            np.cumsum(self.segment_durations), 0, 0.)
        self.duration = self.segment_times[-1]

        self.rabi_rates = np.sqrt(np.sum(self.segments[:, 0:2]**2, axis=1))

        self.maximum_rabi_rate = np.amax(self.rabi_rates)
        self.maximum_detuning = np.amax(np.abs(self.segments[:, 2]))
        self.maximum_amplitude = np.amax(self.amplitudes)
        self.minimum_duration = np.amin(self.segment_durations)
        self.maximum_duration = np.amax(self.segment_durations)

        if self.maximum_rabi_rate > UPPER_BOUND_RABI_RATE:
            raise ArgumentsValueError(
                'Maximum rabi rate of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_RABI_RATE),
                {'segments': self.segments},
                extras={'maximum_rabi_rate': self.maximum_rabi_rate})

        if self.maximum_detuning > UPPER_BOUND_DETUNING_RATE:
            raise ArgumentsValueError(
                'Maximum detuning of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_DETUNING_RATE),
                {'segments': self.segments},
                extras={'maximum_detuning': self.maximum_detuning})
        if self.maximum_duration > UPPER_BOUND_DURATION:
            raise ArgumentsValueError(
                'Maximum duration of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_DURATION),
                {'segments': self.segments},
                extras={'maximum_duration': self.maximum_duration})
        if self.minimum_duration < LOWER_BOUND_DURATION:
            raise ArgumentsValueError(
                'Minimum duration of segments must be larger than the lower bound: '
                + str(LOWER_BOUND_DURATION),
                {'segments': self.segments},
                extras={'minimum_duration'})

    def _qctrl_expanded_export_content(self, file_type, coordinates):

        """Private method to prepare the content to be saved in Q-CTRL expanded format

        Parameters
        ----------
        file_type : str, optional
            One of 'csv' or 'json'; defaults to 'csv'.
        coordinates : str, optional
            Indicates the co-ordinate system requested. Must be one of
            'cylindrical', 'cartesian' or 'polar'; defaults to 'cylindrical'

        Returns
        -------
        list or dict
            Based on file_type; list if 'csv', dict if 'json'
        """

        control_info = None
        if coordinates == CARTESIAN:

            if file_type == CSV:

                control_info = list()
                control_info.append('amplitude_x,amplitude_y,detuning,duration,maximum_rabi_rate')
                for segment_idx in range(self.segments.shape[0]):
                    control_info.append('{},{},{},{},{}'.format(
                        self.segments[segment_idx, 0] / self.maximum_rabi_rate,
                        self.segments[segment_idx, 1] / self.maximum_rabi_rate,
                        self.segments[segment_idx, 2],
                        self.segments[segment_idx, 3],
                        self.maximum_rabi_rate
                    ))
            else:
                control_info = dict()
                if self.name is not None:
                    control_info['name'] = self.name
                control_info['maximum_rabi_rate'] = self.maximum_rabi_rate
                control_info['amplitude_x'] = list(self.segments[:, 0]/self.maximum_rabi_rate)
                control_info['amplitude_y'] = list(self.segments[:, 1] / self.maximum_rabi_rate)
                control_info['detuning'] = list(self.segments[:, 2])
                control_info['duration'] = list(self.segments[:, 3])

        else:

            if file_type == CSV:
                control_info = list()
                control_info.append('rabi_rate,azimuthal_angle,detuning,duration,maximum_rabi_rate')
                for segment_idx in range(self.segments.shape[0]):
                    control_info.append('{},{},{},{},{}'.format(
                        self.rabi_rates[segment_idx]/self.maximum_rabi_rate,
                        np.arctan2(self.segments[segment_idx, 1],
                                   self.segments[segment_idx, 0]),
                        self.segments[segment_idx, 2],
                        self.segments[segment_idx, 3],
                        self.maximum_rabi_rate
                    ))

            else:
                control_info = dict()
                if self.name is not None:
                    control_info['name'] = self.name
                control_info['maximum_rabi_rate'] = self.maximum_rabi_rate
                control_info['rabi_rates'] = list(self.rabi_rates / self.maximum_rabi_rate)
                control_info['azimuthal_angles'] = list(np.arctan2(
                    self.segments[:, 1], self.segments[:, 0]))
                control_info['detuning'] = list(self.segments[:, 2])
                control_info['duration'] = list(self.segments[:, 3])

        return control_info

    def _export_to_qctrl_expanded_format(self, filename=None,
                                         file_type=CSV,
                                         coordinates=CYLINDRICAL):

        """Private method to save control in qctrl_expanded_format

        Parameters
        ----------
        filename : str, optional
            Name and path of the file to save the control into.
            Defaults to None
        file_type : str, optional
            One of 'CSV' or 'JSON'; defaults to 'CSV'.
        coordinates : str, optional
            Indicates the co-ordinate system requested. Must be one of
            'Cylindrical', 'Cartesian'; defaults to 'Cylindrical'
        """

        control_info = self._qctrl_expanded_export_content(file_type=file_type,
                                                           coordinates=coordinates)
        if file_type == CSV:
            with open(filename, 'wt') as handle:

                control_info = '\n'.join(control_info)
                handle.write(control_info)
        else:
            with open(filename, 'wt') as handle:
                json.dump(control_info, handle, sort_keys=True, indent=4)

    def export_to_file(self, filename=None,
                       file_format=QCTRL_EXPANDED,
                       file_type=CSV,
                       coordinates=CYLINDRICAL):

        """Prepares and saves the driven control in a file.

        Parameters
        ----------
        filename : str, optional
            Name and path of the file to save the control into.
            Defaults to None
        file_format : str
            Specified file format for saving the control. Defaults to
            'Q-CTRL expanded'; Currently it does not support any other format.
            For detail of the `Q-CTRL Expanded Format` consult
            `Q-CTRL Control Data Format
            <https://docs.q-ctrl.com/output-data-formats#q-ctrl-hardware>` _.
        file_type : str, optional
            One of 'CSV' or 'JSON'; defaults to 'CSV'.
        coordinates : str, optional
            Indicates the co-ordinate system requested. Must be one of
            'Cylindrical', 'Cartesian'; defaults to 'Cylindrical'

        References
        ----------
        `Q-CTRL Control Data Format
            <https://docs.q-ctrl.com/output-data-formats#q-ctrl-hardware>` _.

        Raises
        ------
        ArgumentsValueError
            Raised if some of the parameters are invalid.
        """

        if filename is None:
            raise ArgumentsValueError('Invalid filename provided.',
                                      {'filename': filename})

        if file_format not in [QCTRL_EXPANDED]:
            raise ArgumentsValueError('Requested file format is not supported. Please use '
                                      'one of {}'.format([QCTRL_EXPANDED]),
                                      {'file_format': file_format})

        if file_type not in [CSV, JSON]:
            raise ArgumentsValueError('Requested file type is not supported. Please use '
                                      'one of {}'.format([CSV, JSON]),
                                      {'file_type': file_type})

        if coordinates not in [CYLINDRICAL, CARTESIAN]:
            raise ArgumentsValueError('Requested coordinate type is not supported. Please use '
                                      'one of {}'.format([CARTESIAN, CYLINDRICAL]),
                                      {'coordinates': coordinates})

        if file_format == QCTRL_EXPANDED:
            self._export_to_qctrl_expanded_format(filename=filename,
                                                  file_type=file_type,
                                                  coordinates=coordinates)

    def get_plot_formatted_arrays(self, coordinates=CARTESIAN, dimensionless_rabi_rate=True):
        """ Gets arrays for plotting a driven control.

        Parameters
        ----------
        dimensionless_rabi_rate: boolean
            If True, calculates the dimensionless values for segments
        coordinates : string
            Indicated the type of segments that need to be transformed can be 'cartesian' or
            'cylindrical'.

        Returns
        -------
        dict
            A dict with keywords depending on the chosen coordinates. For 'cylindrical', we have
            'rabi_rate', 'azimuthal_angle', 'detuning' and 'times', and for 'cartesian' we have
            'amplitude_x', 'amplitude_y', 'detuning' and 'times'.

        Notes
        -----
        The plot data can have repeated times and for amplitudes, because it is expected
        that these coordinates are to be used with plotting software that 'joins the dots' with
        linear lines between each coordinate. The time array gives the x values for all the
        amplitude arrays, which give the y values.

        Raises
        ------
        ArgumentsValueError
            Raised when an argument is invalid.
        """

        plot_segments = self.get_transformed_segments(
            coordinates=CARTESIAN, dimensionless_rabi_rate=dimensionless_rabi_rate)

        plot_data = get_plot_data_from_segments(plot_segments)

        if coordinates == CARTESIAN:
            (x_amplitudes, y_amplitudes, detunings, times) = plot_data
            plot_dictionary = {
                'amplitude_x': x_amplitudes,
                'amplitude_y': y_amplitudes,
                'detuning': detunings,
                'times': times
            }
        elif coordinates == CYLINDRICAL:
            (x_plot, y_plot, detunings, times) = plot_data
            x_plot[np.equal(x_plot, -0.0)] = 0.
            y_plot[np.equal(y_plot, -0.0)] = 0.
            azimuthal_angles_plot = np.arctan2(y_plot, x_plot)
            amplitudes_plot = np.sqrt(np.abs(x_plot**2 + y_plot**2))
            plot_dictionary = {
                'rabi_rate': amplitudes_plot,
                'azimuthal_angle': azimuthal_angles_plot,
                'detuning': detunings,
                'times': times
            }
        else:
            raise ArgumentsValueError(
                'Unsupported coordinates provided: ',
                arguments={'coordinates': coordinates})

        return plot_dictionary

    def get_transformed_segments(self, coordinates=CARTESIAN, dimensionless_rabi_rate=True):
        """
        Function that transforms standard dimension-full segments of the
        driven control into dimensionless segments

        Parameters
        ----------
        coordinates : string
            Indicated the type of segments that need to be transformed can be 'cartesian' or
            'cylindrical' or 'polar'.
        dimensionless_rabi_rate : boolean
            If True, calculates the dimensionless segments

        Returns
        -------
        numpy.ndarray
            if dimensionless is True, returns the dimensionless segments of the driven control;
            otherwise returns the segments

        Raises
        ------
        ArgumentsValueError
            Raised when an argument is invalid.
        """
        dimensionless_rabi_rate = bool(dimensionless_rabi_rate)
        transformed_segments = self.segments.copy()

        if dimensionless_rabi_rate:
            transformed_segments[:, 0:2] = transformed_segments[:, 0:2] / self.maximum_rabi_rate

        if coordinates == CARTESIAN:
            pass
        elif coordinates == CYLINDRICAL:

            temp_amplitudes = np.sqrt(np.abs(
                transformed_segments[:, 0]**2 + transformed_segments[:, 1]**2))
            temp_azimuthal_angles = np.arctan2(
                transformed_segments[:, 1], transformed_segments[:, 0])
            transformed_segments = np.stack(
                (temp_amplitudes,
                 temp_azimuthal_angles,
                 transformed_segments[:, 2],
                 transformed_segments[:, 3]),
                axis=1)
        else:
            raise ArgumentsValueError('Unsupported coordinates provided: ',
                                      arguments={'coordinates': coordinates})

        return transformed_segments

    def __str__(self):
        """Prepares a friendly string format for a Driven Control
        """
        driven_control_string = list()

        # Specify the number of decimals for pretty string representation
        decimals_format_str = '{:+.3f}'

        if self.name is not None:
            driven_control_string.append('{}:'.format(self.name))

        # Format amplitudes
        for i, axis in enumerate('XYZ'):
            pretty_amplitudes_str = ', '.join(
                [decimals_format_str.format(amplitude/np.pi).rstrip('0').rstrip('.')
                 for amplitude in self.segments[:, i]]
            )
            driven_control_string.append(
                '{} Amplitudes: [{}] x pi'.format(axis, pretty_amplitudes_str)
            )
        # Format durations
        total_duration = np.sum(self.segments[:, 3])
        pretty_durations_str = ','.join(
            [decimals_format_str.format(duration/total_duration).rstrip('0').rstrip('.')
             for duration in self.segments[:, 3]]
        )
        driven_control_string.append(
            'Durations:    [{}] x {}s'.format(pretty_durations_str, str(total_duration))
        )

        driven_control_string = '\n'.join(driven_control_string)

        return driven_control_string



if __name__ == '__main__':
    pass
