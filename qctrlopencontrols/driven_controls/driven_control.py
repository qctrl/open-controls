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
Driven control module.
"""
from __future__ import annotations

import csv
import json
from typing import (
    Any,
    Optional,
)

import numpy as np

from ..utils import (
    Coordinate,
    FileFormat,
    FileType,
    check_arguments,
    create_repr_from_attributes,
)


class DrivenControl:
    r"""
    A piecewise-constant driven control for a single qubit.

    Parameters
    ----------
    durations : np.ndarray
        The durations :math:`\{\delta t_n\}` for each segment, in units of seconds. Every element
        must be positive. Represented as a 1D array of length :math:`N`, where :math:`N` is number
        of segments.
    rabi_rates : np.ndarray, optional
        The Rabi rates :math:`\{\Omega_n\}` for each segment, in units of radians per second. Every
        element must be nonnegative. Represented as a 1D array of length :math:`N`, where :math:`N`
        is number of segments. You can omit this field if the Rabi rate is zero on all segments.
    azimuthal_angles : np.ndarray, optional
        The azimuthal angles :math:`\{\phi_n\}` for each segment. Represented as a 1D array of
        length :math:`N`, where :math:`N` is number of segments. You can omit this field if the
        azimuthal angle is zero on all segments.
    detunings : np.ndarray, optional
        The detunings :math:`\{\Delta_n\}` for each segment, in units of radians per second.
        Represented as a 1D array of length :math:`N`, where :math:`N` is number of segments. You
        can omit this field if the detuning is zero on all segments.
    name : string, optional
        An optional string to name the control. Defaults to ``None``.

    Notes
    -----
    This class represents a control for a single driven qubit with Hamiltonian:

    .. math::

        H(t) = \frac{1}{2}\left(\Omega(t) e^{i\phi(t)} \sigma_- +
                                \Omega(t) e^{-i\phi(t)}\sigma_+\right) +
               \frac{1}{2}\Delta(t)\sigma_z,

    where :math:`\Omega(t)` is the Rabi rate, :math:`\phi(t)` is the azimuthal angle (or drive
    phase), :math:`\Delta(t)` is the detuning, :math:`\sigma_\pm = (\sigma_x \mp i\sigma_y)/2`,
    and :math:`\sigma_k` are the Pauli matrices.

    The controls are piecewise-constant, meaning :math:`\Omega(t)=\Omega_n` for
    :math:`t_{n-1}\leq t<t_n`, where :math:`t_0=0` and :math:`t_n=t_{n-1}+\delta t_n` (and similarly
    for :math:`\phi(t)` and :math:`\Delta(t)`).

    For each segment of the control, the constant Hamiltonian effects unitary time evolution of the
    form:

    .. math::

        U_n = \exp\left[-i\frac{\theta_n}{2} (\mathbf u_n\cdot\boldsymbol \sigma)\right],

    where :math:`\theta_n = \sqrt{\Omega_n^2+\Delta_n^2}\delta t_n`,
    :math:`\mathbf u_n` is the unit vector in the direction
    :math:`(\Omega_n\cos\phi_n, \Omega_n\sin\phi_n, \Delta_n)`, and
    :math:`\boldsymbol\sigma=(\sigma_x, \sigma_y, \sigma_z)`. This unitary time evolution
    corresponds to a rotation of the Bloch sphere of an angle :math:`\theta_n` about the axis
    :math:`\mathbf u_n`.
    """

    def __init__(
        self,
        durations: np.ndarray,
        rabi_rates: Optional[np.ndarray] = None,
        azimuthal_angles: Optional[np.ndarray] = None,
        detunings: Optional[np.ndarray] = None,
        name: Optional[str] = None,
    ):

        self.name = name

        durations = np.asarray(durations, dtype=float)

        # check if all the durations are greater than zero
        check_arguments(
            all(durations > 0),
            "Duration of driven control segments must all be positive.",
            {"durations": durations},
        )

        # check if all non-None inputs have the same length
        input_lengths = {
            np.array(v).size
            for v in [rabi_rates, azimuthal_angles, detunings, durations]
            if v is not None
        }

        check_arguments(
            len(input_lengths) == 1,
            "If set, Rabi rates, azimuthal angles, detunings and durations "
            "must be of same length",
            {
                "rabi_rates": rabi_rates,
                "azimuthal_angles": azimuthal_angles,
                "detunings": detunings,
                "durations": durations,
            },
        )

        duration_count = len(durations)

        if rabi_rates is None:
            rabi_rates = np.zeros(duration_count)
        if azimuthal_angles is None:
            azimuthal_angles = np.zeros(duration_count)
        if detunings is None:
            detunings = np.zeros(duration_count)

        # for backward compatibility as these variable could be list
        rabi_rates = np.asarray(rabi_rates, dtype=float)
        azimuthal_angles = np.asarray(azimuthal_angles, dtype=float)
        detunings = np.asarray(detunings, dtype=float)

        # check if all the rabi_rates are nonnegative
        check_arguments(
            all(rabi_rates >= 0.0),
            "All Rabi rates must be nonnegative.",
            {"rabi_rates": rabi_rates},
        )

        self.rabi_rates = rabi_rates
        self.azimuthal_angles = azimuthal_angles
        self.detunings = detunings
        self.durations = durations

    @property
    def number_of_segments(self) -> int:
        """
        Returns the number of segments.

        Returns
        -------
        int
            The number of segments in the driven control, :math:`N`.
        """

        return self.rabi_rates.shape[0]

    @property
    def maximum_rabi_rate(self) -> float:
        r"""
        Returns the maximum Rabi rate of the control.

        Returns
        -------
        float
            The maximum Rabi rate of the control, :math:`\max_n \Omega_n`.
        """

        return np.amax(self.rabi_rates)

    @property
    def maximum_detuning(self) -> float:
        r"""
        Returns the maximum detuning of the control.

        Returns
        -------
        float
            The maximum detuning of the control, :math:`\max_n \Delta_n`.
        """
        return np.amax(self.detunings)

    @property
    def amplitude_x(self) -> np.ndarray:
        r"""
        Returns the x-amplitude.

        Returns
        -------
        np.ndarray
            The x-amplitude of each segment, :math:`\{\Omega_n \cos \phi_n\}`.
        """

        return self.rabi_rates * np.cos(self.azimuthal_angles)

    @property
    def amplitude_y(self) -> np.ndarray:
        r"""
        Returns the y-amplitude.

        Returns
        -------
        np.ndarray
            The y-amplitude of each segment, :math:`\{\Omega_n \sin \phi_n\}`.
        """

        return self.rabi_rates * np.sin(self.azimuthal_angles)

    @property
    def angles(self) -> np.ndarray:
        r"""
        Returns the Bloch sphere rotation angles.

        Returns
        -------
        np.ndarray
            The total Bloch sphere rotation angles on each segment,
            :math:`\left\{\sqrt{\Omega_n^2+\Delta_n^2}\delta t_n\right\}`.
        """

        amplitudes = np.sqrt(
            self.amplitude_x**2 + self.amplitude_y**2 + self.detunings**2
        )
        angles = amplitudes * self.durations

        return angles

    @property
    def directions(self) -> np.ndarray:
        r"""
        Returns the Bloch sphere rotation directions.

        Returns
        -------
        np.ndarray
            The Bloch sphere rotation direction on each segment,
            :math:`\{\mathbf v_n/\|\mathbf v_n\|\}`, where
            :math:`\mathbf v_n=(\Omega_n\cos\phi_n, \Omega_n\sin\phi_n, \Delta_n)`.
        """
        amplitudes = np.sqrt(
            self.amplitude_x**2 + self.amplitude_y**2 + self.detunings**2
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
    def times(self) -> np.ndarray:
        r"""
        Returns the boundary times of the control segments.

        Returns
        ------
        np.ndarray
            The boundary times of the control segments, :math:`\{t_n\}` (starting with
            :math:`t_0=0`).
        """

        return np.insert(np.cumsum(self.durations), 0, 0.0)

    @property
    def maximum_duration(self) -> float:
        r"""
        Returns the duration of the longest control segment.

        Returns
        -------
        float
            The duration of the longest control segment, :math:`\max_n \delta t_n`.
        """

        return np.amax(self.durations)

    @property
    def minimum_duration(self) -> float:
        r"""
        Returns the duration of the shortest control segment.

        Returns
        -------
        float
            The duration of the shortest control segment, :math:`\min_n \delta t_n`.
        """

        return np.amin(self.durations)

    @property
    def duration(self) -> float:
        r"""
        Returns the total duration of the control.

        Returns
        -------
        float
            The total duration of the control, :math:`t_N=\sum_n \delta t_n`.
        """

        return np.sum(self.durations)

    def resample(self, time_step: float, name: Optional[str] = None) -> "DrivenControl":
        r"""
        Returns a new driven control obtained by resampling this control.

        Parameters
        ----------
        time_step : float
            The time step to use for resampling, :math:`\delta t`.
        name : str, optional
            The name for the new control. Defaults to ``None``.

        Returns
        -------
        DrivenControl
            A new driven control, sampled at the specified rate. The durations of the new control
            are all equal to :math:`\delta t`. The total duration of the new control might be
            slightly larger than the original duration, if the time step doesn't exactly divide the
            original duration.
        """
        check_arguments(
            time_step > 0, "Time step must be positive.", {"time_step": time_step}
        )
        check_arguments(
            time_step <= self.duration,
            "Time step must be less than or equal to the original duration.",
            {"time_step": time_step},
            {"duration": self.duration},
        )

        count = int(np.ceil(self.duration / time_step))
        durations = np.repeat(time_step, count)
        times = np.arange(count) * time_step

        indices = np.digitize(times, bins=np.cumsum(self.durations))

        return DrivenControl(
            durations,
            self.rabi_rates[indices],
            self.azimuthal_angles[indices],
            self.detunings[indices],
            name,
        )

    def _qctrl_expanded_export_content(self, coordinates: str) -> dict[str, Any]:
        """
        Prepare the content to be saved in Q-CTRL expanded format.

        Parameters
        ----------
        coordinates : str, optional
            Indicates the co-ordinate system requested. Must be
            'cylindrical'or 'cartesian'. Defaults to 'cylindrical'.

        Returns
        -------
        dict
            A dictionary containing the information of the control.
        """

        control_info = {
            "maximum_rabi_rate": self.maximum_rabi_rate,
            "detuning": list(self.detunings),
            "duration": list(self.durations),
        }

        if self.name is not None:
            control_info["name"] = self.name

        if coordinates == Coordinate.CARTESIAN.value:
            control_info["amplitude_x"] = list(
                self.amplitude_x / self.maximum_rabi_rate
            )
            control_info["amplitude_y"] = list(
                self.amplitude_y / self.maximum_rabi_rate
            )
        else:
            control_info["rabi_rates"] = list(self.rabi_rates / self.maximum_rabi_rate)
            control_info["azimuthal_angles"] = list(self.azimuthal_angles)

        return control_info

    def _export_to_qctrl_expanded_format(
        self,
        filename,
        file_type=FileType.CSV.value,
        coordinates=Coordinate.CYLINDRICAL.value,
    ):
        """
        Saves control in qctrl_expanded_format.

        Parameters
        ----------
        filename : str
            Name and path of the file to save the control into.
        file_type : str, optional
            One of 'CSV' or 'JSON'; defaults to 'CSV'.
        coordinates : str, optional
            Indicates the co-ordinate system requested. Must be one of
            'cylindrical', 'cartesian'; defaults to 'cylindrical'
        """

        control_info = self._qctrl_expanded_export_content(coordinates=coordinates)
        if file_type == FileType.CSV.value:
            _ = control_info.pop("name")
            control_info["maximum_rabi_rate"] = [
                self.maximum_rabi_rate
            ] * self.number_of_segments
            field_names = sorted(control_info.keys())

            # note that the newline parameter here is necessary
            # see details at https://docs.python.org/3/library/csv.html#id3
            with open(filename, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=field_names)
                writer.writeheader()
                for index in range(self.number_of_segments):
                    writer.writerow(
                        {name: control_info[name][index] for name in field_names}
                    )
        else:
            with open(filename, "wt", encoding="utf-8") as handle:
                json.dump(control_info, handle, sort_keys=True, indent=4)

    def export_to_file(
        self,
        filename,
        file_format=FileFormat.QCTRL.value,
        file_type=FileType.CSV.value,
        coordinates=Coordinate.CYLINDRICAL.value,
    ):
        """
        Prepares and saves the driven control in a file.

        Parameters
        ----------
        filename : str
            Name and path of the file to save the control into.
        file_format : str, optional
            Specified file format for saving the control. Defaults to 'Q-CTRL expanded'. Currently
            does not support any other format. For details of the Q-CTRL expanded format, see Notes.
        file_type : str, optional
            One of 'CSV' or 'JSON'. Defaults to 'CSV'.
        coordinates : str, optional
            The coordinate system in which to save the control. Must be 'cylindrical' or
            'cartesian'. Defaults to 'cylindrical'.

        Notes
        -----
        The Q-CTRL expanded format is designed for direct integration of control solutions into
        experimental hardware. The format represents controls as vectors defined for the relevant
        operators sampled in time (corresponding to the segmentation of the Rabi rate, azimuthal
        angle, and detuning).

        The exact data format depends on the file type and coordinate system. In all cases, the data
        contain four lists of real floating point numbers. Each list has the same length, and the
        :math:`n`'th element of each list describes the :math:`n`'th segment of the driven control.

        For Cartesian coordinates, the four lists are X-amplitude, Y-amplitude, detuning, and
        duration. The maximum Rabi rate is also included in the data, and the X-amplitude and
        Y-amplitude are normalized to that maximum Rabi rate.

        For cylindrical coordinates, the four lists are Rabi rate, azimuthal angle, detuning, and
        duration. The maximum Rabi rate is also included in the data, and the Rabi rate is
        normalized to that maximum Rabi rate.

        For CSV, the data are formatted as five columns, with one row of titles, followed by
        :math:`N` rows of data. The first four columns contain the relevant Cartesian or cylindrical
        data. The fifth column contains the maximum Rabi rate, and has the same value in each row.

        For JSON, the data are formatted as a single object (dictionary) with four array fields, a
        "maximum_rabi_rate" field giving the maximum Rabi rate, and optionally a "name" field giving
        the `name` of the control.

        For example, the CSV cylindrical representation of a control with two segments would be::

            rabi_rate,azimuthal_angle,detuning,duration,maximum_rabi_rate
            0.8,1.57,3000000.,0.000001,10000000
            1.0,3.14,-3000000.,0.000002,10000000

        The JSON Cartesian representation of the same control would be::

            {
                "name": "a custom control",
                "maximum_rabi_rate": 10000000,
                "amplitude_x": [0.0,-1.0],
                "amplitude_y": [0.8,0.0],
                "detuning": [3000000.0,-3000000.0],
                "duration": [0.000001,0.000002],
            }
        """
        _file_types = [v.value for v in FileType]
        _file_formats = [v.value for v in FileFormat]
        _coordinate_systems = [v.value for v in Coordinate]

        check_arguments(
            file_format in _file_formats,
            "Requested file format is not supported. Please use "
            f"one of {_file_formats}",
            {"file_format": file_format},
        )

        check_arguments(
            file_type in _file_types,
            "Requested file type is not supported. Please use " f"one of {_file_types}",
            {"file_type": file_type},
        )

        check_arguments(
            coordinates in _coordinate_systems,
            "Requested coordinate type is not supported. Please use "
            f"one of {_coordinate_systems}",
            {"coordinates": coordinates},
        )

        if file_format == FileFormat.QCTRL.value:
            self._export_to_qctrl_expanded_format(
                filename=filename, file_type=file_type, coordinates=coordinates
            )

    def export(
        self, coordinates=Coordinate.CYLINDRICAL.value, dimensionless_rabi_rate=True
    ) -> dict[str, Any]:

        """
        Returns a dictionary formatted for plotting using the ``qctrl-visualizer`` package.

        Parameters
        ----------
        coordinates: string, optional
            Indicates whether the Rabi frequency should be plotted in terms of its
            'cylindrical' or 'cartesian' components. Defaults to 'cylindrical'.
        dimensionless_rabi_rate: boolean, optional
            If ``True``, normalizes the Rabi rate so that its largest absolute value is 1. Defaults
            to ``True``.

        Returns
        -------
        dict
            Dictionary with plot data that can be used by the `plot_controls`
            method of the ``qctrl-visualizer`` package. It has keywords 'Rabi rate'
            and 'Detuning' for 'cylindrical' coordinates and 'X amplitude', 'Y amplitude',
            and 'Detuning' for 'cartesian' coordinates.
        """

        check_arguments(
            coordinates in [v.value for v in Coordinate],
            "Unsupported coordinates provided: ",
            {"coordinates": coordinates},
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
        """
        Prepares a friendly string format for a Driven Control.
        """
        driven_control = []

        if self.name is not None:
            driven_control.append(f"{self.name}:")

        pretty_rabi_rates = ",".join(
            [
                str(rabi_rate / self.maximum_rabi_rate)
                if self.maximum_rabi_rate != 0
                else "0"
                for rabi_rate in self.rabi_rates
            ]
        )

        pretty_azimuthal_angles = ",".join(
            [str(azimuthal_angle / np.pi) for azimuthal_angle in self.azimuthal_angles]
        )

        pretty_detuning = ",".join(
            [
                str(detuning / self.maximum_detuning)
                if self.maximum_detuning != 0
                else "0"
                for detuning in self.detunings
            ]
        )

        pretty_durations = ",".join(
            [str(duration / self.duration) for duration in self.durations]
        )

        driven_control.append(
            f"Rabi Rates = [{pretty_rabi_rates}] × {self.maximum_rabi_rate}"
        )
        driven_control.append(f"Azimuthal Angles = [{pretty_azimuthal_angles}] × pi")
        driven_control.append(
            f"Detunings = [{pretty_detuning}] × {self.maximum_detuning}"
        )
        driven_control.append(f"Durations = [{pretty_durations}] × {self.duration}")
        driven_control_string = "\n".join(driven_control)

        return driven_control_string

    def __repr__(self):
        """
        Returns a string representation for the object.

        The returned string looks like a valid Python expression that could be used to recreate
        the object, including default arguments.

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
