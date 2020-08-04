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
Driven control module.
"""
import json
from typing import Optional

import numpy as np

from ..driven_controls import (
    LOWER_BOUND_DURATION,
    UPPER_BOUND_DETUNING_RATE,
    UPPER_BOUND_DURATION,
    UPPER_BOUND_RABI_RATE,
    UPPER_BOUND_SEGMENTS,
)
from ..exceptions import ArgumentsValueError
from ..utils import (
    Coordinate,
    FileFormat,
    FileType,
    check_arguments,
    create_repr_from_attributes,
)


class DrivenControl:
    """
    Creates a driven control. A driven is a set of segments made up of amplitude vectors
    and corresponding durations.

    Parameters
    ----------
    rabi_rates : numpy.ndarray, optional
        1-D array of size nx1 where n is number of segments;
        Each entry is the rabi rate for the segment. Defaults to None.
    azimuthal_angles : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the azimuthal angle for the segment; Defaults to None.
    detunings : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the detuning angle for the segment; Defaults to None.
    durations : numpy.ndarray, optional
        1-D array of size nx1 where n is the number of segments;
        Each entry is the duration of the segment (in seconds); Defaults to None.
    name : string, optional
        An optional string to name the driven control. Defaults to None.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    def __init__(
        self,
        rabi_rates: Optional[np.ndarray] = None,
        azimuthal_angles: Optional[np.ndarray] = None,
        detunings: Optional[np.ndarray] = None,
        durations: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ):

        self.name = name

        # set default values if all inputs are ``None``
        if all(v is None for v in [rabi_rates, azimuthal_angles, detunings, durations]):
            rabi_rates = np.array([np.pi])
            azimuthal_angles = np.array([0.0])
            detunings = np.array([0.0])
            durations = np.array([1.0])

        # check if all non-None inputs have the same length
        input_lengths = {
            np.array(v).size
            for v in [rabi_rates, azimuthal_angles, detunings, durations]
            if v is not None
        }

        check_arguments(
            len(input_lengths) == 1,
            "Rabi rates, Azimuthal angles, Detunings and Durations "
            "must be of same length",
            {
                "rabi_rates": rabi_rates,
                "azimuthal_angles": azimuthal_angles,
                "detunings": detunings,
                "durations": durations,
            },
        )

        input_length = input_lengths.pop()

        if rabi_rates is None:
            rabi_rates = np.zeros(input_length)
        if azimuthal_angles is None:
            azimuthal_angles = np.zeros(input_length)
        if detunings is None:
            detunings = np.zeros(input_length)
        if durations is None:
            durations = np.ones(input_length)

        self.rabi_rates = np.array(rabi_rates, dtype=np.float).flatten()
        self.azimuthal_angles = np.array(azimuthal_angles, dtype=np.float).flatten()
        self.detunings = np.array(detunings, dtype=np.float).flatten()
        self.durations = np.array(durations, dtype=np.float).flatten()

        # check if all the rabi_rates are greater than zero
        if np.any(self.rabi_rates < 0.0):
            raise ArgumentsValueError(
                "All rabi rates must be greater than zero.",
                {"rabi_rates": rabi_rates},
                extras={
                    "azimuthal_angles": azimuthal_angles,
                    "detunings": detunings,
                    "durations": durations,
                },
            )

        # check if all the durations are greater than zero
        if np.any(self.durations <= 0):
            raise ArgumentsValueError(
                "Duration of driven control segments must all be greater"
                + " than zero.",
                {"durations": self.durations},
            )

        if self.number_of_segments > UPPER_BOUND_SEGMENTS:
            raise ArgumentsValueError(
                "The number of segments must be smaller than the upper bound:"
                + str(UPPER_BOUND_SEGMENTS),
                {"number_of_segments": self.number_of_segments},
            )

        if self.maximum_rabi_rate > UPPER_BOUND_RABI_RATE:
            raise ArgumentsValueError(
                "Maximum rabi rate of segments must be smaller than the upper bound: "
                + str(UPPER_BOUND_RABI_RATE),
                {"maximum_rabi_rate": self.maximum_rabi_rate},
            )

        if self.maximum_detuning > UPPER_BOUND_DETUNING_RATE:
            raise ArgumentsValueError(
                "Maximum detuning of segments must be smaller than the upper bound: "
                + str(UPPER_BOUND_DETUNING_RATE),
                {"maximum_detuning": self.maximum_detuning},
            )
        if self.maximum_duration > UPPER_BOUND_DURATION:
            raise ArgumentsValueError(
                "Maximum duration of segments must be smaller than the upper bound: "
                + str(UPPER_BOUND_DURATION),
                {"maximum_duration": self.maximum_duration},
            )
        if self.minimum_duration < LOWER_BOUND_DURATION:
            raise ArgumentsValueError(
                "Minimum duration of segments must be larger than the lower bound: "
                + str(LOWER_BOUND_DURATION),
                {"minimum_duration": self.minimum_duration},
            )

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

        amplitudes = np.sqrt(
            self.amplitude_x ** 2 + self.amplitude_y ** 2 + self.detunings ** 2
        )
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
        amplitudes = np.sqrt(
            self.amplitude_x ** 2 + self.amplitude_y ** 2 + self.detunings ** 2
        )

        # Reduces tolerance of the comparison to zero in case the units chosen
        # make the amplitudes very small, but never allows it to be higher than the
        # default atol value of 1e-8
        tolerance = min(1e-20 * np.max(amplitudes), 1e-8)

        safe_amplitudes = np.where(
            np.isclose(amplitudes, 0, atol=tolerance), 1.0, amplitudes
        )

        normalized_amplitude_x = self.amplitude_x / safe_amplitudes
        normalized_amplitude_y = self.amplitude_y / safe_amplitudes
        normalized_detunings = self.detunings / safe_amplitudes

        directions = np.hstack(
            (
                normalized_amplitude_x[:, np.newaxis],
                normalized_amplitude_y[:, np.newaxis],
                normalized_detunings[:, np.newaxis],
            )
        )

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

        return np.insert(np.cumsum(self.durations), 0, 0.0)

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
            One of 'CSV' or 'JSON'; defaults to 'CSV'.
        coordinates : str, optional
            Indicates the co-ordinate system requested. Must be one of
            'cylindrical', 'cartesian' or 'polar'; defaults to 'cylindrical'

        Returns
        -------
        list or dict
            Based on file_type; list if 'CSV', dict if 'JSON'
        """
        control_info = None
        amplitude_x = self.amplitude_x
        amplitude_y = self.amplitude_y
        if coordinates == Coordinate.CYLINDRICAL.value:
            if file_type == FileType.CSV.value:
                control_info = list()
                control_info.append(
                    "amplitude_x,amplitude_y,detuning,duration,maximum_rabi_rate"
                )
                for segment_idx in range(self.number_of_segments):
                    control_info.append(
                        "{},{},{},{},{}".format(
                            amplitude_x[segment_idx],
                            amplitude_y[segment_idx],
                            self.detunings[segment_idx],
                            self.durations[segment_idx],
                            self.maximum_rabi_rate,
                        )
                    )
            else:
                control_info = dict()
                if self.name is not None:
                    control_info["name"] = self.name
                control_info["maximum_rabi_rate"] = self.maximum_rabi_rate
                control_info["amplitude_x"] = list(amplitude_x)
                control_info["amplitude_y"] = list(amplitude_y)
                control_info["detuning"] = list(self.detunings)
                control_info["duration"] = list(self.durations)

        else:

            if file_type == FileType.CSV.value:
                control_info = list()
                control_info.append(
                    "rabi_rate,azimuthal_angle,detuning,duration,maximum_rabi_rate"
                )
                for segment_idx in range(self.number_of_segments):
                    control_info.append(
                        "{},{},{},{},{}".format(
                            self.rabi_rates[segment_idx] / self.maximum_rabi_rate,
                            np.arctan2(
                                amplitude_y[segment_idx], amplitude_x[segment_idx]
                            ),
                            self.detunings[segment_idx],
                            self.durations[segment_idx],
                            self.maximum_rabi_rate,
                        )
                    )

            else:
                control_info = dict()
                if self.name is not None:
                    control_info["name"] = self.name
                control_info["maximum_rabi_rate"] = self.maximum_rabi_rate
                control_info["rabi_rates"] = list(
                    self.rabi_rates / self.maximum_rabi_rate
                )
                control_info["azimuthal_angles"] = list(
                    np.arctan2(amplitude_y, amplitude_x)
                )
                control_info["detuning"] = list(self.detunings)
                control_info["duration"] = list(self.durations)

        return control_info

    def _export_to_qctrl_expanded_format(
        self,
        filename=None,
        file_type=FileType.CSV.value,
        coordinates=Coordinate.CYLINDRICAL.value,
    ):
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
            'cylindrical', 'cartesian'; defaults to 'cylindrical'
        """

        control_info = self._qctrl_expanded_export_content(
            file_type=file_type, coordinates=coordinates
        )
        if file_type == FileType.CSV.value:
            with open(filename, "wt") as handle:

                control_info = "\n".join(control_info)
                handle.write(control_info)
        else:
            with open(filename, "wt") as handle:
                json.dump(control_info, handle, sort_keys=True, indent=4)

    def export_to_file(
        self,
        filename=None,
        file_format=FileFormat.QCTRL.value,
        file_type=FileType.CSV.value,
        coordinates=Coordinate.CYLINDRICAL.value,
    ):
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
            <https://docs.q-ctrl.com/wiki/output-data-formats#q-ctrl-hardware>` _.
        file_type : str, optional
            One of 'CSV' or 'JSON'; defaults to 'CSV'.
        coordinates : str, optional
            Indicates the co-ordinate system requested. Must be one of
            'cylindrical', 'cartesian'; defaults to 'cylindrical'

        References
        ----------
        `Q-CTRL Control Data Format
        <https://docs.q-ctrl.com/wiki/output-data-formats#q-ctrl-hardware>` _.

        Raises
        ------
        ArgumentsValueError
            Raised if some of the parameters are invalid.
        """
        _file_types = [v.value for v in FileType]
        _file_formats = [v.value for v in FileFormat]
        _coordinate_systems = [v.value for v in Coordinate]

        if filename is None:
            raise ArgumentsValueError(
                "Invalid filename provided.", {"filename": filename}
            )

        if file_format not in _file_formats:
            raise ArgumentsValueError(
                "Requested file format is not supported. Please use "
                "one of {}".format(_file_formats),
                {"file_format": file_format},
            )

        if file_type not in _file_types:
            raise ArgumentsValueError(
                "Requested file type is not supported. Please use "
                "one of {}".format(_file_types),
                {"file_type": file_type},
            )

        if coordinates not in _coordinate_systems:
            raise ArgumentsValueError(
                "Requested coordinate type is not supported. Please use "
                "one of {}".format(_coordinate_systems),
                {"coordinates": coordinates},
            )

        if file_format == FileFormat.QCTRL.value:
            self._export_to_qctrl_expanded_format(
                filename=filename, file_type=file_type, coordinates=coordinates
            )

    def export(
        self, coordinates=Coordinate.CYLINDRICAL.value, dimensionless_rabi_rate=True
    ):

        """ Returns a dictionary formatted for plotting using the qctrl-visualizer package.

        Parameters
        ----------
        dimensionless_rabi_rate: boolean
            If True, normalizes the Rabi rate so that its largest absolute value is 1.
        coordinates: string
            Indicates whether the Rabi frequency should be plotted in terms of its
            'cylindrical' or 'cartesian' components.

        Returns
        -------
        dict
            Dictionary with plot data that can be used by the plot_controls
            method of the qctrl-visualizer package. It has keywords 'Rabi rate'
            and 'Detuning' for 'cylindrical' coordinates and 'X amplitude', 'Y amplitude',
            and 'Detuning' for 'cartesian' coordinates.

        Raises
        ------
        ArgumentsValueError
            Raised when an argument is invalid.
        """

        if coordinates not in [v.value for v in Coordinate]:
            raise ArgumentsValueError(
                "Unsupported coordinates provided: ",
                arguments={"coordinates": coordinates},
            )

        if dimensionless_rabi_rate:
            normalizer = self.maximum_rabi_rate
        else:
            normalizer = 1

        plot_dictionary = {}

        plot_x = self.amplitude_x / normalizer
        plot_y = self.amplitude_y / normalizer
        plot_r = self.rabi_rates / normalizer
        plot_theta = self.azimuthal_angles
        plot_durations = self.durations
        plot_detunings = self.detunings

        if coordinates == Coordinate.CARTESIAN.value:
            plot_dictionary["X amplitude"] = [
                {"value": v, "duration": t} for v, t in zip(plot_x, plot_durations)
            ]
            plot_dictionary["Y amplitude"] = [
                {"value": v, "duration": t} for v, t in zip(plot_y, plot_durations)
            ]

        if coordinates == Coordinate.CYLINDRICAL.value:
            plot_dictionary["Rabi rate"] = [
                {"value": r * np.exp(1.0j * theta), "duration": t}
                for r, theta, t in zip(plot_r, plot_theta, plot_durations)
            ]

        plot_dictionary["Detuning"] = [
            {"value": v, "duration": t} for v, t in zip(plot_detunings, plot_durations)
        ]

        return plot_dictionary

    def __str__(self):
        """Prepares a friendly string format for a Driven Control
        """
        driven_control_string = list()

        if self.name is not None:
            driven_control_string.append("{}:".format(self.name))

        pretty_rabi_rates = [
            str(rabi_rate / self.maximum_rabi_rate)
            if self.maximum_rabi_rate != 0
            else "0"
            for rabi_rate in list(self.rabi_rates)
        ]
        pretty_rabi_rates = ",".join(pretty_rabi_rates)
        pretty_azimuthal_angles = [
            str(azimuthal_angle / np.pi) for azimuthal_angle in self.azimuthal_angles
        ]
        pretty_azimuthal_angles = ",".join(pretty_azimuthal_angles)
        pretty_detuning = [
            str(detuning / self.maximum_detuning) if self.maximum_detuning != 0 else "0"
            for detuning in list(self.detunings)
        ]
        pretty_detuning = ",".join(pretty_detuning)

        pretty_durations = [
            str(duration / self.duration) for duration in self.durations
        ]
        pretty_durations = ",".join(pretty_durations)

        driven_control_string.append(
            "Rabi Rates = [{}] x {}".format(pretty_rabi_rates, self.maximum_rabi_rate)
        )
        driven_control_string.append(
            "Azimuthal Angles = [{}] x pi".format(pretty_azimuthal_angles)
        )
        driven_control_string.append(
            "Detunings = [{}] x {}".format(pretty_detuning, self.maximum_detuning)
        )
        driven_control_string.append(
            "Durations = [{}] x {}".format(pretty_durations, self.duration)
        )
        driven_control_string = "\n".join(driven_control_string)

        return driven_control_string

    def __repr__(self):
        """Returns a string representation for the object. The returned string looks like a valid
        Python expression that could be used to recreate the object, including default arguments.

        Returns
        -------
        str
            String representation of the object including the values of the arguments.
        """

        attributes = [
            "rabi_rates",
            "azimuthal_angles",
            "detunings",
            "durations",
            "name",
        ]

        return create_repr_from_attributes(self, attributes)
