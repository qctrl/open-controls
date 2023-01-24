# Copyright 2023 Q-CTRL
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
Dynamical decoupling module.
"""

from __future__ import annotations

from typing import (
    Any,
    Optional,
)

import numpy as np

from ..driven_controls.driven_control import DrivenControl
from ..exceptions import ArgumentsValueError
from ..utils import (
    Coordinate,
    FileFormat,
    FileType,
    check_arguments,
    create_repr_from_attributes,
)


class DynamicDecouplingSequence:
    r"""
    Creates a dynamic decoupling sequence.

    Parameters
    ----------
    duration : float
        The total time in seconds for the sequence :math:`\tau`.
    offsets : np.ndarray
        The times offsets :math:`\{t_j\}` in seconds for the center of pulses.
    rabi_rotations : np.ndarray
        The Rabi rotation :math:`\omega_j` at each time offset :math:`t_j`.
    azimuthal_angles : np.ndarray
        The azimuthal angle :math:`\phi_j` at each time offset :math:`t_j`.
    detuning_rotations : np.ndarray
        The detuning rotation :math:`\delta_j` at each time offset :math:`t_j`.
    name : str, optional
        Name of the sequence. Defaults to None.

    Notes
    -----
    Dynamical decoupling sequence (DDS) is canonically defined as a series of
    :math:`n`-instantaneous unitary operations, often :math:`\pi`-pulses, executed
    at time offsets :math:`\{t_j\}_{j=1}^n` over the time interval with a total
    duration :math:`\tau`. The :math:`j`-th operation applied at time
    :math:`t_j` can be parameterized as

    .. math::
        U_j = \exp\left[-\frac{i}{2}(\omega_j \cos \phi_j \sigma_x + \omega_j\sin \phi_j\sigma_y
        + \delta_j\sigma_z)\right] \;,

    Note that in practice all DDSs typically have a :math:`X_{\pi/2}` operation at the start
    :math:`t = 0` and end :math:`t = \tau` of the sequence. This is because it is assumed that the
    qubit is initially in the state :math:`|0\rangle` and a superposition needs to be created and
    removed to make the qubit sensitive to dephasing.
    """

    def __init__(
        self,
        duration: float,
        offsets: np.ndarray,
        rabi_rotations: np.ndarray,
        azimuthal_angles: np.ndarray,
        detuning_rotations: np.ndarray,
        name: Optional[str] = None,
    ):

        check_arguments(
            duration > 0,
            "Sequence duration must be above zero.",
            {"duration": duration},
        )

        offsets = np.asarray(offsets)
        check_arguments(
            np.all((offsets >= 0) & (duration >= offsets)),
            "Offsets for dynamic decoupling sequence must be between 0 and the sequence "
            "duration (inclusive). ",
            {"offsets": offsets, "duration": duration},
        )

        rabi_rotations = np.asarray(rabi_rotations, dtype=float)
        check_arguments(
            np.all(rabi_rotations >= 0),
            "Rabi rotations must be nonnegative.",
            {"rabi_rotations": rabi_rotations},
        )

        _offset_count = len(offsets)

        check_arguments(
            len(rabi_rotations) == _offset_count,
            "rabi rotations must have the same length as offsets. ",
            {"offsets": offsets, "rabi_rotations": rabi_rotations},
        )

        check_arguments(
            len(azimuthal_angles) == _offset_count,
            "azimuthal angles must have the same length as offsets. ",
            {"offsets": offsets, "azimuthal_angles": azimuthal_angles},
        )

        check_arguments(
            len(detuning_rotations) == _offset_count,
            "detuning rotations must have the same length as offsets. ",
            {"offsets": offsets, "detuning_rotations": detuning_rotations},
        )

        self.duration = duration
        self.offsets = offsets
        self.rabi_rotations = rabi_rotations
        self.azimuthal_angles = np.asarray(azimuthal_angles, dtype=float)
        self.detuning_rotations = np.asarray(detuning_rotations, dtype=float)
        self.name = name

    def export(self) -> dict[str, Any]:
        """
        Returns a dictionary for plotting using the qctrl-visualizer package.

        Returns
        -------
        dict
            Dictionary with plot data that can be used by the plot_sequences
            method of the qctrl-visualizer package. It has keywords 'Rabi'
            and 'Detuning'.
        """

        return {
            "Rabi": [
                {"rotation": rabi * np.exp(1.0j * theta), "offset": offset}
                for rabi, theta, offset in zip(
                    self.rabi_rotations, self.azimuthal_angles, self.offsets
                )
            ],
            "Detuning": [
                {"rotation": rotation, "offset": offset}
                for rotation, offset in zip(self.detuning_rotations, self.offsets)
            ],
        }

    def __repr__(self):
        """
        Returns a string representation for the object.

        The returned string looks like a valid
        Python expression that could be used to recreate the object, including default arguments.

        Returns
        -------
        str
            String representation of the object including the values of the arguments.
        """

        attributes = [
            "duration",
            "offsets",
            "rabi_rotations",
            "azimuthal_angles",
            "detuning_rotations",
            "name",
        ]

        return create_repr_from_attributes(self, attributes)

    def __str__(self):
        """
        Prepares a friendly string format for a dynamical decoupling sequence.
        """

        def _array_to_str(arr: np.ndarray) -> str:
            """
            Converts elements of an array to a string.
            [1, 2] -> "1, 2"
            """
            return ", ".join(arr.astype(str))

        sequence_string = []

        if self.name is not None:
            sequence_string.append(f"{self.name}:")

        sequence_string.append(f"Duration = {self.duration}")

        sequence_string.append(
            f"Offsets = [{_array_to_str(self.offsets / self.duration)}] × {self.duration}"
        )

        sequence_string.append(
            f"Rabi Rotations = [{_array_to_str(self.rabi_rotations / np.pi)}] × pi"
        )

        sequence_string.append(
            f"Azimuthal Angles = [{_array_to_str(self.azimuthal_angles / np.pi)}] × pi"
        )

        sequence_string.append(
            f"Detuning Rotations = [{_array_to_str(self.detuning_rotations / np.pi)}] × pi"
        )

        return "\n".join(sequence_string)

    def export_to_file(
        self,
        filename: str,
        file_format: str = FileFormat.QCTRL.value,
        file_type: str = FileType.CSV.value,
        coordinates: str = Coordinate.CYLINDRICAL.value,
        maximum_rabi_rate: float = 2 * np.pi,
        maximum_detuning_rate: float = 2 * np.pi,
    ) -> None:
        r"""
        Prepares and saves the dynamical decoupling sequence in a file.

        Parameters
        ----------
        filename : str
            Name and path of the file to save the control into.
        file_format : str
            Specified file format for saving the control. Defaults to
            'Q-CTRL expanded'. Currently it does not support any other format.
            For detail of the `Q-CTRL Expanded Format` consult
            :py:meth:`DrivenControl.export_to_file`.
        file_type : str, optional
            One of 'CSV' or 'JSON'. Defaults to 'CSV'.
        coordinates : str, optional
            Indicates the coordinate system requested. Must be one of
            'cylindrical' or 'cartesian'. Defaults to 'cylindrical'.
        maximum_rabi_rate : float, optional
            Maximum Rabi rate. Defaults to :math:`2\pi`.
        maximum_detuning_rate : float, optional
            Maximum detuning rate. Defaults to :math:`2\pi`.

        Raises
        ------
        ArgumentsValueError
            Raised if some of the parameters are invalid.

        Notes
        -----
        The sequence is converted to a driven control using the maximum Rabi and detuning
        rate. The driven control is then exported.
        """

        convert_dds_to_driven_control(
            dynamic_decoupling_sequence=self,
            maximum_rabi_rate=maximum_rabi_rate,
            maximum_detuning_rate=maximum_detuning_rate,
            name=self.name,
        ).export_to_file(
            filename=filename,
            file_format=file_format,
            file_type=file_type,
            coordinates=coordinates,
        )


def convert_dds_to_driven_control(
    dynamic_decoupling_sequence: DynamicDecouplingSequence,
    maximum_rabi_rate: float,
    maximum_detuning_rate: float,
    minimum_segment_duration: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a Driven Control based on the supplied DDS and other relevant information.

    Currently, pulses that simultaneously contain Rabi and detuning rotations are not
    supported.

    Parameters
    ----------
    dynamic_decoupling_sequence : DynamicDecouplingSequence
        The base DDS. Its offsets should be sorted in ascending order in time.
    maximum_rabi_rate : float
        Maximum Rabi rate.
    maximum_detuning_rate : float
        Maximum detuning rate.
    minimum_segment_duration : float, optional
        If set, further restricts the duration of every segment of the Driven Controls.
        Defaults to 0, in which case it does not affect the duration of the pulses.
        Must be greater than or equal to 0, if set.
    name : str, optional
        Name of the sequence. Defaults to None.

    Returns
    -------
    DrivenControls
        The Driven Control that contains the segments
        corresponding to the Dynamic Decoupling Sequence operation.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid or a valid driven control cannot be
        created from the sequence parameters, maximum Rabi rate and maximum detuning
        rate provided.

    Notes
    -----
    Driven pulse is defined as a sequence of control segments. Each segment performs
    an operation (rotation around one or more axes). While the dynamic decoupling
    sequence operation contains ideal instant operations, the maximum Rabi (detuning) rate
    defines a minimum time required to perform a given rotation operation. Therefore, each
    operation in sequence is converted to a flat-topped control segment with a finite duration.
    Each offset is taken as the mid-point of the control segment and the width of the
    segment is determined by (rotation/max_rabi(detuning)_rate).

    If the sequence contains operations at either of the extreme ends
    :math:`\tau_0=0` and :math:`\tau_{n+1}=\tau`(duration of the sequence), there
    will be segments outside the boundary (segments starting before :math:`t<0`
    or finishing after the sequence duration :math:`t>\tau`). In these cases, the segments
    on either of the extreme ends are shifted appropriately so that their start/end time
    falls entirely within the duration of the sequence.

    Moreover, a check is made to make sure the resulting control segments are non-overlapping.

    If appropriate control segments cannot be created, the conversion process raises
    an ArgumentsValueError.
    """

    check_arguments(
        maximum_detuning_rate > 0,
        "Maximum detuning rate must be positive.",
        {"maximum_detuning_rate": maximum_detuning_rate},
    )
    check_arguments(
        maximum_rabi_rate > 0,
        "Maximum Rabi rate must be positive.",
        {"maximum_rabi_rate": maximum_rabi_rate},
    )

    check_arguments(
        minimum_segment_duration >= 0,
        "Minimum segment duration must be greater than or equal to 0.",
        {"minimum_segment_duration": minimum_segment_duration},
    )

    sequence_duration = dynamic_decoupling_sequence.duration
    offsets = dynamic_decoupling_sequence.offsets
    rabi_rotations = dynamic_decoupling_sequence.rabi_rotations
    azimuthal_angles = dynamic_decoupling_sequence.azimuthal_angles
    detuning_rotations = dynamic_decoupling_sequence.detuning_rotations

    # check if all Rabi rotations are valid (i.e. have positive values)
    check_arguments(
        np.all(rabi_rotations >= 0.0),
        "Sequence contains negative values for Rabi rotations.",
        {"dynamic_decoupling_sequence": dynamic_decoupling_sequence},
    )

    # check for valid operation
    check_arguments(
        _check_valid_operation(
            rabi_rotations=rabi_rotations, detuning_rotations=detuning_rotations
        ),
        "Sequence operation includes Rabi rotation and "
        "detuning rotation at the same instance.",
        {"dynamic_decoupling_sequence": dynamic_decoupling_sequence},
        extras={
            "maximum_rabi_rate": maximum_rabi_rate,
            "maximum_detuning_rate": maximum_detuning_rate,
        },
    )

    if offsets.size == 0:
        offsets = np.array([0, sequence_duration])
        rabi_rotations = np.array([0, 0])
        azimuthal_angles = np.array([0, 0])
        detuning_rotations = np.array([0, 0])

    if offsets[0] != 0:
        offsets = np.append([0], offsets)
        rabi_rotations = np.append([0], rabi_rotations)
        azimuthal_angles = np.append([0], azimuthal_angles)
        detuning_rotations = np.append([0], detuning_rotations)
    if offsets[-1] != sequence_duration:
        offsets = np.append(offsets, [sequence_duration])
        rabi_rotations = np.append(rabi_rotations, [0])
        azimuthal_angles = np.append(azimuthal_angles, [0])
        detuning_rotations = np.append(detuning_rotations, [0])

    # check that the offsets are correctly sorted in time
    if any(np.diff(offsets) <= 0.0):
        raise ArgumentsValueError(
            "Pulse timing could not be properly deduced from "
            "the sequence offsets. Make sure all offsets are "
            "in increasing order.",
            {"dynamic_decoupling_sequence": dynamic_decoupling_sequence},
            extras={"offsets": offsets},
        )

    offsets = offsets[np.newaxis, :]
    rabi_rotations = rabi_rotations[np.newaxis, :]
    azimuthal_angles = azimuthal_angles[np.newaxis, :]
    detuning_rotations = detuning_rotations[np.newaxis, :]

    operations = np.concatenate(
        (offsets, rabi_rotations, azimuthal_angles, detuning_rotations), axis=0
    )

    pulse_mid_points = operations[0, :]

    pulse_start_ends = np.zeros((operations.shape[1], 2))
    for op_idx in range(operations.shape[1]):
        # Pulses that cause no rotations can have 0 duration
        half_pulse_duration = 0.0

        if not np.isclose(operations[1, op_idx], 0.0):  # Rabi rotation
            half_pulse_duration = 0.5 * max(
                operations[1, op_idx] / maximum_rabi_rate, minimum_segment_duration
            )
        elif not np.isclose(operations[3, op_idx], 0.0):  # Detuning rotation
            half_pulse_duration = 0.5 * max(
                np.abs(operations[3, op_idx]) / maximum_detuning_rate,
                minimum_segment_duration,
            )

        pulse_start_ends[op_idx, 0] = pulse_mid_points[op_idx] - half_pulse_duration
        pulse_start_ends[op_idx, 1] = pulse_mid_points[op_idx] + half_pulse_duration

    # check if any of the pulses have gone outside the time limit [0, sequence_duration]
    # if yes, adjust the segment timing
    if pulse_start_ends[0, 0] < 0.0:
        translation = 0.0 - (pulse_start_ends[0, 0])
        pulse_start_ends[0, :] = pulse_start_ends[0, :] + translation

    if pulse_start_ends[-1, 1] > sequence_duration:
        translation = pulse_start_ends[-1, 1] - sequence_duration
        pulse_start_ends[-1, :] = pulse_start_ends[-1, :] - translation

    # check if the minimum_segment_duration is respected in the gaps between the pulses
    # as minimum_segment_duration >= 0, this also excludes overlaps
    gap_durations = pulse_start_ends[1:, 0] - pulse_start_ends[:-1, 1]
    if not np.all(
        np.logical_or(
            np.greater(gap_durations, minimum_segment_duration),
            np.isclose(gap_durations, minimum_segment_duration),
        )
    ):
        raise ArgumentsValueError(
            "Distance between pulses does not respect minimum_segment_duration. "
            "Try decreasing the minimum_segment_duration or increasing "
            "the maximum_rabi_rate or the maximum_detuning_rate.",
            {
                "dynamic_decoupling_sequence": dynamic_decoupling_sequence,
                "maximum_rabi_rate": maximum_rabi_rate,
                "maximum_detuning_rate": maximum_detuning_rate,
                "minimum_segment_duration": minimum_segment_duration,
            },
            extras={
                "deduced_pulse_start_timing": pulse_start_ends[:, 0],
                "deduced_pulse_end_timing": pulse_start_ends[:, 1],
                "gap_durations": gap_durations,
            },
        )

    if np.allclose(pulse_start_ends, 0.0):
        # the original sequence should be a free evolution
        return DrivenControl(
            rabi_rates=np.array([0.0]),
            azimuthal_angles=np.array([0.0]),
            detunings=np.array([0.0]),
            durations=np.array([sequence_duration]),
            name=name,
        )

    control_rabi_rates = np.zeros((operations.shape[1] * 2,))
    control_azimuthal_angles = np.zeros((operations.shape[1] * 2,))
    control_detunings = np.zeros((operations.shape[1] * 2,))
    control_durations = np.zeros((operations.shape[1] * 2,))

    pulse_segment_idx = 0
    for op_idx in range(0, operations.shape[1]):
        pulse_width = pulse_start_ends[op_idx, 1] - pulse_start_ends[op_idx, 0]
        control_durations[pulse_segment_idx] = pulse_width

        if pulse_width > 0.0:
            if not np.isclose(operations[1, op_idx], 0.0):  # Rabi rotation
                control_rabi_rates[pulse_segment_idx] = (
                    operations[1, op_idx] / pulse_width
                )
                control_azimuthal_angles[pulse_segment_idx] = operations[2, op_idx]
            elif not np.isclose(operations[3, op_idx], 0.0):  # Detuning rotation
                control_detunings[pulse_segment_idx] = (
                    operations[3, op_idx] / pulse_width
                )

        if op_idx != (operations.shape[1] - 1):
            control_rabi_rates[pulse_segment_idx + 1] = 0.0
            control_azimuthal_angles[pulse_segment_idx + 1] = 0.0
            control_detunings[pulse_segment_idx + 1] = 0.0
            control_durations[pulse_segment_idx + 1] = (
                pulse_start_ends[op_idx + 1, 0] - pulse_start_ends[op_idx, 1]
            )

        pulse_segment_idx += 2

    # almost there; let us check if there is any segments with durations = 0
    control_rabi_rates = control_rabi_rates[control_durations > 0.0]
    control_azimuthal_angles = control_azimuthal_angles[control_durations > 0.0]
    control_detunings = control_detunings[control_durations > 0.0]
    control_durations = control_durations[control_durations > 0.0]

    return DrivenControl(
        rabi_rates=control_rabi_rates,
        azimuthal_angles=control_azimuthal_angles,
        detunings=control_detunings,
        durations=control_durations,
        name=name,
    )


def _check_valid_operation(
    rabi_rotations: np.ndarray, detuning_rotations: np.ndarray
) -> bool:
    """
    Private method to check if there is a rabi_rotation and detuning rotation at the same
    offset.
    """

    rabi_rotation_index = set(np.where(rabi_rotations > 0.0)[0])
    detuning_rotation_index = set(np.where(detuning_rotations > 0.0)[0])

    return not rabi_rotation_index.intersection(detuning_rotation_index)
