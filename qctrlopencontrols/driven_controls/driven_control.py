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
===============================
driven_controls.driven_controls
===============================
"""
import json
import numpy as np

from qctrlopencontrols.exceptions import ArgumentsValueError
from qctrlopencontrols.base import create_repr_from_attributes

from qctrlopencontrols.globals import (
    QCTRL_EXPANDED, CSV, JSON, CARTESIAN, CYLINDRICAL)

from .constants import (
    UPPER_BOUND_SEGMENTS, UPPER_BOUND_RABI_RATE, UPPER_BOUND_DETUNING_RATE,
    UPPER_BOUND_DURATION, LOWER_BOUND_DURATION)


class DrivenControl(object):   #pylint: disable=too-few-public-methods
    """
    Creates a driven control. A driven is a set of segments made up of amplitude vectors
    and corresponding durations.

    Parameters
    ----------
    rabi_rates : numpy.ndarray, optional
        1-D array of size nx1 where n is number of segments;
        Each entry is the rabi rate for the segment. Defaults to None
    azimuthal_angles : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the azimuthal angle for the segment; Defaults to None
    detunings : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the detuning angle for the segment; Defaults to None
    durations : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the duration of the segment (in seconds); Defaults to None
    name : string, optional
        An optional string to name the driven control. Defaults to None.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    def __init__(self,
                 rabi_rates=None,
                 azimuthal_angles=None,
                 detunings=None,
                 durations=None,
                 name=None):

        self.name = name
        if self.name is not None:
            self.name = str(self.name)

        check_none_values = [(rabi_rates is None), (azimuthal_angles is None),
                             (detunings is None), (durations is None)]
        all_are_none = all(value is True for value in check_none_values)
        if all_are_none:
            rabi_rates = np.array([np.pi])
            azimuthal_angles = np.array([0.])
            detunings = np.array([0.])
            durations = np.array([1.])
        else:
            # some may be None while others are not
            input_array_lengths = []
            if not check_none_values[0]:
                rabi_rates = np.array(rabi_rates, dtype=np.float).reshape((-1,))
                input_array_lengths.append(rabi_rates.shape[0])

            if not check_none_values[1]:
                azimuthal_angles = np.array(azimuthal_angles, dtype=np.float).reshape((-1,))
                input_array_lengths.append(len(azimuthal_angles))

            if not check_none_values[2]:
                detunings = np.array(detunings, dtype=np.float).reshape((-1,))
                input_array_lengths.append(len(detunings))

            if not check_none_values[3]:
                durations = np.array(durations, dtype=np.float).reshape((-1,))
                input_array_lengths.append(len(durations))

            # check all valid array lengths are equal
            if max(input_array_lengths) != min(input_array_lengths):
                raise ArgumentsValueError('Rabi rates, Azimuthal angles, Detunings and Durations '
                                          'must be of same length',
                                          {'rabi_rates': rabi_rates,
                                           'azimuthal_angles': azimuthal_angles,
                                           'detunings': detunings,
                                           'durations': durations})

            valid_input_length = max(input_array_lengths)
            if check_none_values[0]:
                rabi_rates = np.zeros((valid_input_length,))
            if check_none_values[1]:
                azimuthal_angles = np.zeros((valid_input_length,))
            if check_none_values[2]:
                detunings = np.zeros((valid_input_length,))
            if check_none_values[3]:
                durations = np.ones((valid_input_length,))

        self.rabi_rates = rabi_rates
        self.azimuthal_angles = azimuthal_angles
        self.detunings = detunings
        self.durations = durations

        # check if all the rabi_rates are greater than zero
        if np.any(rabi_rates < 0.):
            raise ArgumentsValueError('All rabi rates must be greater than zero.',
                                      {'rabi_rates': rabi_rates},
                                      extras={
                                          'azimuthal_angles': azimuthal_angles,
                                          'detunings': detunings,
                                          'durations': durations})

        # check if all the durations are greater than zero
        if np.any(durations <= 0):
            raise ArgumentsValueError('Duration of driven control segments must all be greater'
                                      + ' than zero.',
                                      {'durations': self.durations})

        if self.number_of_segments > UPPER_BOUND_SEGMENTS:
            raise ArgumentsValueError(
                'The number of segments must be smaller than the upper bound:'
                + str(UPPER_BOUND_SEGMENTS),
                {'number_of_segments': self.number_of_segments})

        if self.maximum_rabi_rate > UPPER_BOUND_RABI_RATE:
            raise ArgumentsValueError(
                'Maximum rabi rate of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_RABI_RATE),
                {'maximum_rabi_rate': self.maximum_rabi_rate})

        if self.maximum_detuning > UPPER_BOUND_DETUNING_RATE:
            raise ArgumentsValueError(
                'Maximum detuning of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_DETUNING_RATE),
                {'maximum_detuning': self.maximum_detuning})
        if self.maximum_duration > UPPER_BOUND_DURATION:
            raise ArgumentsValueError(
                'Maximum duration of segments must be smaller than the upper bound: '
                + str(UPPER_BOUND_DURATION),
                {'maximum_duration': self.maximum_duration})
        if self.minimum_duration < LOWER_BOUND_DURATION:
            raise ArgumentsValueError(
                'Minimum duration of segments must be larger than the lower bound: '
                + str(LOWER_BOUND_DURATION),
                {'minimum_duration': self.minimum_duration})

    @property
    def number_of_segments(self):

        """Returns the number of segments

        Returns
        -------
        int
            The number of segments in the driven control
        """

        return self.rabi_rates.shape[0]

    @property
    def maximum_rabi_rate(self):
        """Returns the maximum rabi rate of the control

        Returns
        -------
        float
            The maximum rabi rate of the control
        """

        return np.amax(self.rabi_rates)

    @property
    def maximum_detuning(self):
        """Returns the maximum detuning of the control

        Returns
        -------
        float
            The maximum detuning of the control
        """
        return np.amax(self.detunings)

    @property
    def amplitude_x(self):
        """Return the X-Amplitude

        Returns
        -------
        numpy.ndarray
            X-Amplitude of each segment
        """

        return self.rabi_rates * np.cos(self.azimuthal_angles)

    @property
    def amplitude_y(self):
        """Return the Y-Amplitude

        Returns
        -------
        numpy.ndarray
            Y-Amplitude of each segment
        """

        return self.rabi_rates * np.sin(self.azimuthal_angles)

    @property
    def angles(self):
        """Returns the angles

        Returns
        -------
        numpy.darray
            Angles as 1-D array of floats
        """

        amplitudes = np.sqrt(self.amplitude_x ** 2 +
                             self.amplitude_y ** 2 +
                             self.detunings ** 2)
        angles = amplitudes * self.durations

        return angles

    @property
    def directions(self):

        """Returns the directions

        Returns
        -------
        numpy.ndarray
            Directions as 1-D array of floats
        """
        amplitudes = np.sqrt(self.amplitude_x ** 2 +
                             self.amplitude_y ** 2 +
                             self.detunings ** 2)
        normalized_amplitude_x = self.amplitude_x/amplitudes
        normalized_amplitude_y = self.amplitude_y/amplitudes
        normalized_detunings = self.detunings/amplitudes

        normalized_amplitudes = np.hstack((normalized_amplitude_x[:, np.newaxis],
                                           normalized_amplitude_y[:, np.newaxis],
                                           normalized_detunings[:, np.newaxis]))

        directions = np.array([normalized_amplitudes if amplitudes[i] != 0. else
                               np.zeros([3, ]) for i in range(self.number_of_segments)])

        return directions

    @property
    def times(self):
        """Returns the time of each segment within the duration
        of the control

        Returns
        ------
        numpy.ndarray
            Segment times as 1-D array of floats
        """

        return np.insert(np.cumsum(self.durations), 0, 0.)

    @property
    def maximum_duration(self):
        """Returns the maximum duration of all the control segments

        Returns
        -------
        float
            The maximum duration of all the control segments
        """

        return np.amax(self.durations)

    @property
    def minimum_duration(self):
        """Returns the minimum duration of all the control segments

        Returns
        -------
        float
            The minimum duration of all the controls segments
        """

        return np.amin(self.durations)

    @property
    def duration(self):
        """Returns the total duration of the control

        Returns
        -------
        float
            Total duration of the control
        """

        return np.sum(self.durations)

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
        amplitude_x = self.amplitude_x
        amplitude_y = self.amplitude_y
        if coordinates == CARTESIAN:
            if file_type == CSV:

                control_info = list()
                control_info.append('amplitude_x,amplitude_y,detuning,duration,maximum_rabi_rate')
                for segment_idx in range(self.number_of_segments):
                    control_info.append('{},{},{},{},{}'.format(
                        amplitude_x[segment_idx],
                        amplitude_y[segment_idx],
                        self.detunings[segment_idx],
                        self.durations[segment_idx],
                        self.maximum_rabi_rate
                    ))
            else:
                control_info = dict()
                if self.name is not None:
                    control_info['name'] = self.name
                control_info['maximum_rabi_rate'] = self.maximum_rabi_rate
                control_info['amplitude_x'] = list(amplitude_x)
                control_info['amplitude_y'] = list(amplitude_y)
                control_info['detuning'] = list(self.detunings)
                control_info['duration'] = list(self.durations)

        else:

            if file_type == CSV:
                control_info = list()
                control_info.append('rabi_rate,azimuthal_angle,detuning,duration,maximum_rabi_rate')
                for segment_idx in range(self.number_of_segments):
                    control_info.append('{},{},{},{},{}'.format(
                        self.rabi_rates[segment_idx]/self.maximum_rabi_rate,
                        np.arctan2(amplitude_y[segment_idx],
                                   amplitude_x[segment_idx]),
                        self.detunings[segment_idx],
                        self.durations[segment_idx],
                        self.maximum_rabi_rate
                    ))

            else:
                control_info = dict()
                if self.name is not None:
                    control_info['name'] = self.name
                control_info['maximum_rabi_rate'] = self.maximum_rabi_rate
                control_info['rabi_rates'] = list(self.rabi_rates / self.maximum_rabi_rate)
                control_info['azimuthal_angles'] = list(np.arctan2(
                    amplitude_y, amplitude_x))
                control_info['detuning'] = list(self.detunings)
                control_info['duration'] = list(self.durations)

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
        if coordinates not in [CARTESIAN, CYLINDRICAL]:
            raise ArgumentsValueError(
                'Unsupported coordinates provided: ',
                arguments={'coordinates': coordinates})

        if dimensionless_rabi_rate:
            normalizer = self.maximum_rabi_rate
        else:
            normalizer = 1

        if coordinates == CARTESIAN:
            control_segments = np.vstack((
                self.amplitude_x / normalizer,
                self.amplitude_y / normalizer,
                self.detunings,
                self.durations)).T
        elif coordinates == CYLINDRICAL:
            control_segments = np.vstack((
                self.rabi_rates / normalizer,
                self.azimuthal_angles,
                self.detunings,
                self.durations)).T

        segment_times = np.insert(np.cumsum(control_segments[:, 3]), 0, 0.)
        plot_time = (segment_times[:, np.newaxis] * np.ones((1, 2))).flatten()
        plot_amplitude_x = control_segments[:, 0]
        plot_amplitude_y = control_segments[:, 1]
        plot_amplitude_z = control_segments[:, 2]

        plot_amplitude_x = np.concatenate(
            ([0.], (plot_amplitude_x[:, np.newaxis] * np.ones((1, 2))).flatten(), [0.]))
        plot_amplitude_y = np.concatenate(
            ([0.], (plot_amplitude_y[:, np.newaxis] * np.ones((1, 2))).flatten(), [0.]))
        plot_amplitude_z = np.concatenate(
            ([0.], (plot_amplitude_z[:, np.newaxis] * np.ones((1, 2))).flatten(), [0.]))

        plot_dictionary = {}
        if coordinates == CARTESIAN:
            plot_dictionary = {
                'amplitudes_x': plot_amplitude_x,
                'amplitudes_y': plot_amplitude_y,
                'detunings': plot_amplitude_z,
                'times': plot_time}

        if coordinates == CYLINDRICAL:

            x_plot = plot_amplitude_x
            y_plot = plot_amplitude_y
            x_plot[np.equal(x_plot, -0.0)] = 0.
            y_plot[np.equal(y_plot, -0.0)] = 0.
            azimuthal_angles_plot = np.arctan2(y_plot, x_plot)
            amplitudes_plot = np.sqrt(np.abs(x_plot**2 + y_plot**2))

            plot_dictionary = {
                'rabi_rates': amplitudes_plot,
                'azimuthal_angles': azimuthal_angles_plot,
                'detunings': plot_amplitude_z,
                'times': plot_time}
        return plot_dictionary

    def __str__(self):
        """Prepares a friendly string format for a Driven Control
        """
        driven_control_string = list()

        if self.name is not None:
            driven_control_string.append('{}:'.format(self.name))

        pretty_rabi_rates = [str(rabi_rate/self.maximum_rabi_rate)
                             if self.maximum_rabi_rate != 0 else '0'
                             for rabi_rate in list(self.rabi_rates)]
        pretty_rabi_rates = ','.join(pretty_rabi_rates)
        pretty_azimuthal_angles = [str(azimuthal_angle/np.pi)
                                   for azimuthal_angle in self.azimuthal_angles]
        pretty_azimuthal_angles = ','.join(pretty_azimuthal_angles)
        pretty_detuning = [str(detuning/self.maximum_detuning)
                           if self.maximum_detuning != 0 else '0'
                           for detuning in list(self.detunings)]
        pretty_detuning = ','.join(pretty_detuning)

        pretty_durations = [str(duration/self.duration) for duration in self.durations]
        pretty_durations = ','.join(pretty_durations)

        driven_control_string.append(
            'Rabi Rates = [{}] x {}'.format(pretty_rabi_rates,
                                            self.maximum_rabi_rate))
        driven_control_string.append(
            'Azimuthal Angles = [{}] x pi'.format(pretty_azimuthal_angles))
        driven_control_string.append(
            'Detunings = [{}] x {}'.format(pretty_detuning,
                                           self.maximum_detuning))
        driven_control_string.append('Durations = [{}] x {}'.format(pretty_durations,
                                                                    self.duration))
        driven_control_string = '\n'.join(driven_control_string)

        return driven_control_string

    def __repr__(self):

        """Returns a string representation for the object. The returned string looks like a valid
        Python expression that could be used to recreate the object, including default arguments.

        Returns
        -------
        str
            String representation of the object including the values of the arguments.
        """

        attributes = {
            'rabi_rates': self.rabi_rates,
            'azimuthal_angles': self.azimuthal_angles,
            'detunings': self.detunings,
            'durations': self.durations,
            'name': self.name
        }

        class_name = '{0.__class__.__name__!s}'.format(self)

        return create_repr_from_attributes(class_name, **attributes)


if __name__ == '__main__':
    pass
