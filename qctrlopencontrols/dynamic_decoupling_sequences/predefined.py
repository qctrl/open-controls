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
========================================
dynamic_decoupling_sequences.predefined
========================================
"""

import numpy as np

from ..dynamic_decoupling_sequences import (
    CARR_PURCELL,
    CARR_PURCELL_MEIBOOM_GILL,
    PERIODIC_SINGLE_AXIS,
    QUADRATIC,
    RAMSEY,
    SPIN_ECHO,
    UHRIG_SINGLE_AXIS,
    WALSH_SINGLE_AXIS,
    X_CONCATENATED,
    XY_CONCATENATED,
)
from ..exceptions import ArgumentsValueError
from .dynamic_decoupling_sequence import DynamicDecouplingSequence


def _add_pre_post_rotations(
    duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
):
    """Adds a pre-post pi.2 rotation at the
    start and end of the sequence.

    The parameters of the pi/2-pulses are chosen in order to cancel out the
    product of the pulses in the DSS, so that its total effect in the
    absence of noise is an identity.

    For a DSS that already produces an identity, this function adds X pi/2-pulses
    in opposite directions, so that they cancel out. If the DDS produces an X
    gate, the X pi/2-pulses will be in the same direction. If the DDS produces
    a Y (Z) gate, the pi/2-pulses are around the Y (Z) axis.

    This function assumes that the sequences only have X, Y, and Z pi-pulses.
    An exception is thrown if that is not the case.

    Parameters
    ----------
    duration: float
        The duration of the sequence
    offsets : numpy.ndarray
        Offsets of the sequence.
    rabi_rotations: numpy.ndarray
        Rabi rotations at each of the offsets.
    azimuthal_angles : numpy.ndarray
        Azimuthal angles at each of the offsets.
    detuning_rotations: numpy.ndarray
        Detuning rotations at each of the offsets

    Returns
    -------
    tuple
        Containing the (offsets, rabi_rotations, azimuthal_angles, detuning_rotations)
        resulting after the addition of pi/2 pulses at the start and end of the sequence.

    Raises
    -----
    ArgumentsValueError
        Raised when sequence does not consist solely of X, Y, and Z pi-pulses.
    """
    # Count the number of X, Y, and Z pi-pulses
    x_pi_pulses = np.count_nonzero(
        np.logical_and.reduce(  # pylint: disable=maybe-no-member
            (
                np.isclose(rabi_rotations, np.pi),
                np.isclose(azimuthal_angles, 0.0),
                np.isclose(detuning_rotations, 0.0),
            )
        )
    )
    y_pi_pulses = np.count_nonzero(
        np.logical_and.reduce(  # pylint: disable=maybe-no-member
            (
                np.isclose(rabi_rotations, np.pi),
                np.isclose(azimuthal_angles, np.pi / 2.0),
                np.isclose(detuning_rotations, 0.0),
            )
        )
    )
    z_pi_pulses = np.count_nonzero(
        np.logical_and.reduce(  # pylint: disable=maybe-no-member
            (
                np.isclose(rabi_rotations, 0.0),
                np.isclose(azimuthal_angles, 0.0),
                np.isclose(detuning_rotations, np.pi),
            )
        )
    )

    # Check if the sequence consists solely of X, Y, and Z pi-pulses
    if len(offsets) != x_pi_pulses + y_pi_pulses + z_pi_pulses:
        raise ArgumentsValueError(
            "Sequence contains pulses that are not X, Y, or Z pi-pulses.",
            {
                "rabi_rotations": rabi_rotations,
                "azimuthal_angles": azimuthal_angles,
                "detuning_rotations": detuning_rotations,
            },
        )

    # The sequence will preserve the state |0> is it has an even number
    # of X and Y pi-pulses
    preserves_10 = (x_pi_pulses + y_pi_pulses) % 2 == 0

    # The sequence will preserve the state |0>+|1> is it has an even number
    # of Y and Z pi-pulses
    preserves_11 = (y_pi_pulses + z_pi_pulses) % 2 == 0

    # When states |0> and |0>+|1> are preserved, the sequence already produces
    # an identity, so that we want the the pi/2-pulses to cancel each other out
    if preserves_10 and preserves_11:
        rabi_value = np.pi / 2
        initial_azimuthal = 0
        final_azimuthal = np.pi
        detuning_value = 0

    # When only state |0>+|1> is not preserved, the sequence results in a Z rotation.
    # In this case, we want both pi/2-pulses to be in the Z direction,
    # so that the remaining rotation is cancelled out
    if preserves_10 and not preserves_11:
        rabi_value = 0
        initial_azimuthal = 0
        final_azimuthal = 0
        detuning_value = np.pi / 2

    # When only state |0> is not preserved, the sequence results in an X rotation.
    # In this case, we want both pi/2-pulses to be in the X direction,
    # so that the remaining rotation is cancelled out
    if not preserves_10 and preserves_11:
        rabi_value = np.pi / 2
        initial_azimuthal = 0
        final_azimuthal = 0
        detuning_value = 0

    # When neither state is preserved, the sequence results in a Y rotation.
    # In this case, we want both pi/2-pulses to be in the Y direction,
    # so that the remaining rotation is cancelled out
    if not preserves_10 and not preserves_11:
        rabi_value = np.pi / 2
        initial_azimuthal = np.pi / 2
        final_azimuthal = np.pi / 2
        detuning_value = 0

    offsets = np.insert(offsets, [0, offsets.shape[0]], [0, duration],)
    rabi_rotations = np.insert(
        rabi_rotations, [0, rabi_rotations.shape[0]], [rabi_value, rabi_value],
    )
    azimuthal_angles = np.insert(
        azimuthal_angles,
        [0, azimuthal_angles.shape[0]],
        [initial_azimuthal, final_azimuthal],
    )
    detuning_rotations = np.insert(
        detuning_rotations,
        [0, detuning_rotations.shape[0]],
        [detuning_value, detuning_value],
    )

    return offsets, rabi_rotations, azimuthal_angles, detuning_rotations


def new_predefined_dds(scheme=SPIN_ECHO, **kwargs):
    """Create a new instance of one of the predefined
    dynamic decoupling sequences

    Parameters
    ----------
    scheme : string
        The name of the sequence; Defaults to 'Spin echo'
        Available options are,
        - 'Ramsey'
        - 'Spin echo',
        - 'Carr-Purcell',
        - 'Carr-Purcell-Meiboom-Gill',
        - 'Uhrig single-axis'
        - 'Periodic single-axis'
        - 'Walsh single-axis'
        - 'Quadratic'
        - 'X concatenated'
        - 'XY concatenated'
    kwargs : dict, optional
        Additional keyword argument to create the sequence

    Returns
    ------
    qctrlopencontrols.dynamic_decoupling_sequences.DynamicDecouplingSequence
        Returns a sequence corresponding to the name

    Raises
    -----
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    if scheme == RAMSEY:
        sequence = _new_ramsey_sequence(**kwargs)
    elif scheme == SPIN_ECHO:
        sequence = _new_spin_echo_sequence(**kwargs)
    elif scheme == CARR_PURCELL:
        sequence = _new_carr_purcell_sequence(**kwargs)
    elif scheme == CARR_PURCELL_MEIBOOM_GILL:
        sequence = _new_carr_purcell_meiboom_gill_sequence(**kwargs)
    elif scheme == UHRIG_SINGLE_AXIS:
        sequence = _new_uhrig_single_axis_sequence(**kwargs)
    elif scheme == PERIODIC_SINGLE_AXIS:
        sequence = _new_periodic_single_axis_sequence(**kwargs)
    elif scheme == WALSH_SINGLE_AXIS:
        sequence = _new_walsh_single_axis_sequence(**kwargs)
    elif scheme == QUADRATIC:
        sequence = _new_quadratic_sequence(**kwargs)
    elif scheme == X_CONCATENATED:
        sequence = _new_x_concatenated_sequence(**kwargs)
    elif scheme == XY_CONCATENATED:
        sequence = _new_xy_concatenated_sequence(**kwargs)
    # Raise an error if the input sequence is not known
    else:
        raise ArgumentsValueError(
            "Unknown predefined sequence scheme. Allowed schemes are: "
            + ", ".join(
                [
                    RAMSEY,
                    SPIN_ECHO,
                    CARR_PURCELL,
                    CARR_PURCELL_MEIBOOM_GILL,
                    UHRIG_SINGLE_AXIS,
                    PERIODIC_SINGLE_AXIS,
                    WALSH_SINGLE_AXIS,
                    QUADRATIC,
                    X_CONCATENATED,
                    XY_CONCATENATED,
                ]
            )
            + ".",
            {"sequence_name": scheme},
        )

    return sequence


def _check_duration(duration):
    """Validates sequence duration
    Parameters
    ----------
    duration : float, optional
        Total duration of the sequence. Defaults to None

    Returns
    -------
    float
        The validated duration

    Raises
    ------
    ArgumentsValueError
        If the duration is negative
    """
    if duration is None:
        duration = 1.0
    if duration <= 0.0:
        raise ArgumentsValueError(
            "Sequence duration must be above zero:", {"duration": duration}
        )
    return duration


def _new_ramsey_sequence(duration=None, pre_post_rotation=False, **kwargs):
    """Ramsey sequence

    Parameters
    ----------
    duration : float, optional
        Total duration of the sequence. Defaults to None
    pre_post_rotation : bool, optional
        If True, a :math:`X_{\\pi.2}` rotation
        is added at the start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamic_decoupling_sequences.DynamicDecouplingSequence
        The Ramsey sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.

    """
    duration = _check_duration(duration)
    offsets = []
    rabi_rotations = []
    azimuthal_angles = []
    detuning_rotations = []

    if pre_post_rotation:
        offsets = duration * np.array([0.0, 1.0])
        rabi_rotations = np.array([np.pi / 2, np.pi / 2])
        azimuthal_angles = np.array([0.0, np.pi])
        detuning_rotations = np.zeros(offsets.shape)

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_spin_echo_sequence(duration=None, pre_post_rotation=False, **kwargs):
    """Spin Echo Sequence.

    Parameters
    ---------
    duration : float, optional
        Total duration of the sequence. Defaults to None
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamic_decoupling_sequences.DynamicDecouplingSequence
        Spin echo sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    duration = _check_duration(duration)
    offsets = duration * np.array([0.5])
    rabi_rotations = np.array([np.pi])
    azimuthal_angles = np.zeros(offsets.shape)
    detuning_rotations = np.zeros(offsets.shape)

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_carr_purcell_sequence(
    duration=None, number_of_offsets=None, pre_post_rotation=False, **kwargs
):
    """Carr-Purcell Sequence.

    Parameters
    ---------
    duration : float, optional
        Total duration of the sequence. Defaults to None
    number_of_offsets : int, optional
        Number of offsets. Defaults to None
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence
        Carr-Purcell sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    duration = _check_duration(duration)
    number_of_offsets = number_of_offsets or 1
    number_of_offsets = int(number_of_offsets)
    if number_of_offsets <= 0.0:
        raise ArgumentsValueError(
            "Number of offsets must be above zero:",
            {"number_of_offsets": number_of_offsets},
        )

    offsets = _carr_purcell_meiboom_gill_offsets(duration, number_of_offsets)

    rabi_rotations = np.zeros(offsets.shape)
    # set all as X_pi
    rabi_rotations[0:] = np.pi
    azimuthal_angles = np.zeros(offsets.shape)
    detuning_rotations = np.zeros(offsets.shape)

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_carr_purcell_meiboom_gill_sequence(
    duration=None, number_of_offsets=None, pre_post_rotation=False, **kwargs
):
    """Carr-Purcell-Meiboom-Gill Sequences.

    Parameters
    ---------
    duration : float
        Total duration of the sequence. Defaults to None
    number_of_offsets : int, optional
        Number of offsets. Defaults to None
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence
        Carr-Purcell-Meiboom-Gill sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    duration = _check_duration(duration)
    number_of_offsets = number_of_offsets or 1
    number_of_offsets = int(number_of_offsets)
    if number_of_offsets <= 0.0:
        raise ArgumentsValueError(
            "Number of offsets must be above zero:",
            {"number_of_offsets": number_of_offsets},
        )

    offsets = _carr_purcell_meiboom_gill_offsets(duration, number_of_offsets)
    rabi_rotations = np.zeros(offsets.shape)
    azimuthal_angles = np.zeros(offsets.shape)

    # set all azimuthal_angles=pi/2, rabi_rotations = pi
    rabi_rotations[0:] = np.pi
    azimuthal_angles[0:] = np.pi / 2
    detuning_rotations = np.zeros(offsets.shape)

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_uhrig_single_axis_sequence(
    duration=None, number_of_offsets=None, pre_post_rotation=False, **kwargs
):
    """Uhrig Single Axis Sequence.

    Parameters
    ---------
    duration : float
        Total duration of the sequence. Defaults to None
    number_of_offsets : int, optional
        Number of offsets. Defaults to None
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence
        Uhrig (single-axis) sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    duration = _check_duration(duration)
    number_of_offsets = number_of_offsets or 1
    number_of_offsets = int(number_of_offsets)
    if number_of_offsets <= 0.0:
        raise ArgumentsValueError(
            "Number of offsets must be above zero:",
            {"number_of_offsets": number_of_offsets},
        )

    offsets = _uhrig_single_axis_offsets(duration, number_of_offsets)
    rabi_rotations = np.zeros(offsets.shape)
    azimuthal_angles = np.zeros(offsets.shape)

    # set all azimuthal_angles=pi/2, rabi_rotations = pi
    rabi_rotations[0:] = np.pi
    azimuthal_angles[0:] = np.pi / 2
    detuning_rotations = np.zeros(offsets.shape)

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_periodic_single_axis_sequence(
    duration=None, number_of_offsets=None, pre_post_rotation=False, **kwargs
):
    """Periodic Single Axis Sequence.

    Parameters
    ---------
    duration : float
        Total duration of the sequence. Defaults to None
    number_of_offsets : int, optional
        Number of offsets. Defaults to None
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence
        Periodic (single-axis) sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    duration = _check_duration(duration)
    number_of_offsets = number_of_offsets or 1
    number_of_offsets = int(number_of_offsets)
    if number_of_offsets <= 0.0:
        raise ArgumentsValueError(
            "Number of offsets must be above zero:",
            {"number_of_offsets": number_of_offsets},
        )

    spacing = 1.0 / (number_of_offsets + 1)
    deltas = [k * spacing for k in range(1, number_of_offsets + 1)]
    deltas = np.array(deltas)
    offsets = duration * deltas
    rabi_rotations = np.zeros(offsets.shape)
    rabi_rotations[0:] = np.pi
    azimuthal_angles = np.zeros(offsets.shape)
    detuning_rotations = np.zeros(offsets.shape)

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_walsh_single_axis_sequence(
    duration=None, paley_order=None, pre_post_rotation=False, **kwargs
):
    """Walsh Single Axis Sequence.

    Parameters
    ---------
    duration : float
        Total duration of the sequence. Defaults to None
    paley_order : int, optional
        Defaults to 1. The paley order of the walsh sequence.
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence
        Walsh (single-axis) sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    duration = _check_duration(duration)
    paley_order = paley_order or 1
    paley_order = int(paley_order)
    if paley_order < 1 or paley_order > 2000:
        raise ArgumentsValueError(
            "Paley order must be between 1 and 2000.", {"paley_order": paley_order}
        )

    hamming_weight = int(np.floor(np.log2(paley_order))) + 1

    samples = 2 ** hamming_weight

    relative_offset = np.arange(1.0 / (2 * samples), 1.0, 1.0 / samples)

    binary_string = np.binary_repr(paley_order)
    binary_order = [int(binary_string[i]) for i in range(hamming_weight)]
    walsh_array = np.ones([samples])
    for i in range(hamming_weight):
        walsh_array *= (
            np.sign(np.sin(2 ** (i + 1) * np.pi * relative_offset))
            ** binary_order[hamming_weight - 1 - i]
        )

    walsh_relative_offsets = []
    for i in range(samples - 1):
        if walsh_array[i] != walsh_array[i + 1]:
            walsh_relative_offsets.append((i + 1) * (1.0 / samples))
    walsh_relative_offsets = np.array(walsh_relative_offsets, dtype=np.float)
    offsets = duration * walsh_relative_offsets
    rabi_rotations = np.zeros(offsets.shape)
    rabi_rotations[0:] = np.pi
    azimuthal_angles = np.zeros(offsets.shape)
    detuning_rotations = np.zeros(offsets.shape)

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_quadratic_sequence(
    duration=None,
    number_inner_offsets=None,
    number_outer_offsets=None,
    pre_post_rotation=False,
    **kwargs
):
    """Quadratic Decoupling Sequence

    Parameters
    ----------
    duration : float, optional
        defaults to None
        The total duration of the sequence
    number_outer_offsets : int, optional
        Number of outer X-pi Pulses. Defaults to None. Not used if number_of_offsets
        is supplied.
    number_inner_offsets : int, optional
        Number of inner Z-pi Pulses. Defaults to None. Not used if number_of_offsets
        is supplied
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence
        Quadratic sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    duration = _check_duration(duration)

    number_inner_offsets = number_inner_offsets or 1
    number_inner_offsets = int(number_inner_offsets)
    if number_inner_offsets <= 0.0:
        raise ArgumentsValueError(
            "Number of offsets of inner pulses must be above zero:",
            {"number_inner_offsets": number_inner_offsets},
            extras={"duration": duration, "number_outer_offsets": number_outer_offsets},
        )

    number_outer_offsets = number_outer_offsets or 1
    number_outer_offsets = int(number_outer_offsets)
    if number_outer_offsets <= 0.0:
        raise ArgumentsValueError(
            "Number of offsets of outer pulses must be above zero:",
            {"number_inner_offsets": number_outer_offsets},
            extras={"duration": duration, "number_inner_offsets": number_inner_offsets},
        )

    outer_offsets = _uhrig_single_axis_offsets(duration, number_outer_offsets)
    outer_offsets = np.insert(outer_offsets, [0, outer_offsets.shape[0]], [0, duration])
    starts = outer_offsets[0:-1]
    ends = outer_offsets[1:]
    inner_durations = ends - starts

    offsets = np.zeros((inner_durations.shape[0], number_inner_offsets + 1))
    for inner_duration_idx in range(inner_durations.shape[0]):
        inn_off = _uhrig_single_axis_offsets(
            inner_durations[inner_duration_idx], number_inner_offsets
        )
        inn_off = inn_off + starts[inner_duration_idx]
        offsets[inner_duration_idx, 0:number_inner_offsets] = inn_off
    offsets[0:number_outer_offsets, -1] = outer_offsets[1:-1]

    rabi_rotations = np.zeros(offsets.shape)
    detuning_rotations = np.zeros(offsets.shape)

    rabi_rotations[0:number_outer_offsets, -1] = np.pi
    detuning_rotations[0 : (number_outer_offsets + 1), 0:number_inner_offsets] = np.pi

    offsets = np.reshape(offsets, (-1,))
    rabi_rotations = np.reshape(rabi_rotations, (-1,))
    detuning_rotations = np.reshape(detuning_rotations, (-1,))

    # remove the last entry corresponding to the duration
    offsets = offsets[0:-1]
    rabi_rotations = rabi_rotations[0:-1]
    detuning_rotations = detuning_rotations[0:-1]
    azimuthal_angles = np.zeros(offsets.shape)

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_x_concatenated_sequence(
    duration=1.0, concatenation_order=None, pre_post_rotation=False, **kwargs
):
    """X-Concatenated Dynamic Decoupling Sequence
    Concatenation of base sequence C(\tau/2)XC(\tau/2)X

    Parameters
    ----------
    duration : float, optional
        defaults to None
        The total duration of the sequence
    concatenation_order : int, optional
        defaults to None
        The number of concatenation of base sequence
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence
        X concatenated sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    duration = _check_duration(duration)

    concatenation_order = concatenation_order or 1
    concatenation_order = int(concatenation_order)
    if concatenation_order <= 0.0:
        raise ArgumentsValueError(
            "Concatenation oder must be above zero:",
            {"concatenation_order": concatenation_order},
            extras={"duration": duration},
        )

    unit_spacing = duration / (2 ** concatenation_order)
    cumulations = _concatenation_x(concatenation_order)

    pos_cum = cumulations * unit_spacing
    pos_cum_sum = np.cumsum(pos_cum)

    values, counts = np.unique(pos_cum_sum, return_counts=True)

    offsets = [values[i] for i in range(counts.shape[0]) if counts[i] % 2 == 0]

    if concatenation_order % 2 == 1:
        offsets = offsets[0:-1]

    offsets = np.array(offsets)
    rabi_rotations = np.zeros(offsets.shape)
    rabi_rotations[0:] = np.pi
    azimuthal_angles = np.zeros(offsets.shape)
    detuning_rotations = np.zeros(offsets.shape)

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _new_xy_concatenated_sequence(
    duration=1.0, concatenation_order=None, pre_post_rotation=False, **kwargs
):
    """XY-Concatenated Dynamic Decoupling Sequence
    Concatenation of base sequence C(\tau/4)XC(\tau/4)YC(\tau/4)XC(\tau/4)Y

    Parameters
    ----------
    duration : float, optional
        defaults to None
        The total duration of the sequence
    concatenation_order : int, optional
        defaults to None
        The number of concatenation of base sequence
    pre_post_rotation : bool, optional
        If True, a :math:`\\pi.2` rotation is added at the
        start and end of the sequence.
    kwargs : dict
        Additional keywords required by
        qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence

    Returns
    -------
    qctrlopencontrols.dynamical_decoupling_sequences.DynamicDecouplingSequence
        XY concatenated sequence

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    duration = _check_duration(duration)

    concatenation_order = concatenation_order or 1
    concatenation_order = int(concatenation_order)
    if concatenation_order <= 0.0:
        raise ArgumentsValueError(
            "Concatenation order must be above zero:",
            {"concatenation_order": concatenation_order},
            extras={"duration": duration},
        )

    unit_spacing = duration / (2 ** (concatenation_order * 2))
    cumulations = _concatenation_xy(concatenation_order)

    rabi_operations = cumulations[cumulations != -2]
    rabi_operations = rabi_operations[rabi_operations != -3]
    rabi_positions = np.zeros(rabi_operations.shape)
    rabi_positions[rabi_operations != -1] = 1
    rabi_positions = rabi_positions * unit_spacing
    rabi_positions = np.cumsum(rabi_positions)

    values, counts = np.unique(rabi_positions, return_counts=True)
    rabi_offsets = [values[i] for i in range(counts.shape[0]) if counts[i] % 2 == 0]

    azimuthal_operations = cumulations[cumulations != -1]
    azimuthal_operations = azimuthal_operations[azimuthal_operations != -3]
    azimuthal_positions = np.zeros(azimuthal_operations.shape)
    azimuthal_positions[azimuthal_operations != -2] = 1
    azimuthal_positions = azimuthal_positions * unit_spacing
    azimuthal_positions = np.cumsum(azimuthal_positions)

    values, counts = np.unique(azimuthal_positions, return_counts=True)
    azimuthal_offsets = [
        values[i] for i in range(counts.shape[0]) if counts[i] % 2 == 0
    ]

    detuning_operations = cumulations[cumulations != -2]
    detuning_operations = detuning_operations[detuning_operations != -1]
    detuning_positions = np.zeros(detuning_operations.shape)
    detuning_positions[detuning_operations != -3] = 1
    detuning_positions = detuning_positions * unit_spacing
    detuning_positions = np.cumsum(detuning_positions)

    values, counts = np.unique(detuning_positions, return_counts=True)
    detuning_offsets = [values[i] for i in range(counts.shape[0]) if counts[i] % 2 == 0]

    # right now we have got all the offset positions separately; now have
    # put then all together

    offsets = np.zeros(
        (len(rabi_offsets) + len(azimuthal_offsets) + len(detuning_offsets),)
    )

    rabi_rotations = np.zeros(offsets.shape)
    azimuthal_angles = np.zeros(offsets.shape)
    detuning_rotations = np.zeros(offsets.shape)

    rabi_idx = 0
    azimuthal_idx = 0

    carr_idx = 0
    while rabi_idx < len(rabi_offsets) and azimuthal_idx < len(azimuthal_offsets):

        if rabi_offsets[rabi_idx] < azimuthal_offsets[azimuthal_idx]:
            rabi_rotations[carr_idx] = np.pi
            offsets[carr_idx] = rabi_offsets[rabi_idx]
            rabi_idx += 1
        else:
            azimuthal_angles[carr_idx] = np.pi / 2
            rabi_rotations[carr_idx] = np.pi
            offsets[carr_idx] = azimuthal_offsets[azimuthal_idx]
            azimuthal_idx += 1
        carr_idx += 1

    if rabi_idx < len(rabi_offsets):

        while rabi_idx < len(rabi_offsets):
            rabi_rotations[carr_idx] = np.pi
            offsets[carr_idx] = rabi_offsets[rabi_idx]
            carr_idx += 1
            rabi_idx += 1
    if azimuthal_idx < len(azimuthal_offsets):
        while azimuthal_idx < len(azimuthal_offsets):
            azimuthal_angles[carr_idx] = np.pi / 2
            rabi_rotations[carr_idx] = np.pi
            offsets[carr_idx] = azimuthal_offsets[azimuthal_idx]
            carr_idx += 1
            azimuthal_idx += 1

    # if there is any z-offset, add those too !!!
    if detuning_offsets:
        z_idx = 0
        for carr_idx, offset in enumerate(offsets):
            if offset > detuning_offsets[z_idx]:
                offsets[carr_idx + 1 :] = offsets[carr_idx:-1]
                rabi_rotations[carr_idx + 1 :] = rabi_rotations[carr_idx:-1]
                azimuthal_angles[carr_idx + 1 :] = azimuthal_angles[carr_idx:-1]
                detuning_rotations[carr_idx] = np.pi
                rabi_rotations[carr_idx] = 0
                azimuthal_angles[carr_idx] = 0
                offsets[carr_idx] = detuning_offsets[z_idx]
                z_idx += 1
            if z_idx >= len(detuning_offsets):
                break

    if pre_post_rotation:
        (
            offsets,
            rabi_rotations,
            azimuthal_angles,
            detuning_rotations,
        ) = _add_pre_post_rotations(
            duration, offsets, rabi_rotations, azimuthal_angles, detuning_rotations
        )

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        **kwargs
    )


def _carr_purcell_meiboom_gill_offsets(duration=1.0, number_of_offsets=1):
    """Offset values for Carr-Purcell_Meiboom-Gill sequence.

    Parameters
    ----------
    duration : float, optional
        Duration of the total sequence; defaults to 1.0
    number_of_offsets : int, optional
        The number of offsets; defaults to 1

    Returns
    ------
    numpy.ndarray
        The offset values
    """

    spacing = 1.0 / number_of_offsets
    start = spacing * 0.5

    # prepare the offsets for delta comb
    deltas = spacing * np.arange(number_of_offsets)
    deltas += start
    offsets = deltas * duration

    return offsets


def _uhrig_single_axis_offsets(duration=1.0, number_of_offsets=1):
    """Offset values for Uhrig Single Axis Sequence.

    Parameters
    ----------
    duration : float, optional
        Duration of the total sequence; defaults to 1.0
    number_of_offsets : int, optional
        The number of offsets; defaults to 1

    Returns
    ------
    numpy.ndarray
        The offset values
    """

    # prepare the offsets for delta comb
    constant = 1.0 / (2 * number_of_offsets + 2)
    deltas = [
        (np.sin(np.pi * k * constant)) ** 2 for k in range(1, number_of_offsets + 1)
    ]
    deltas = np.array(deltas)
    offsets = duration * deltas

    return offsets


def _concatenation_x(concatenation_sequence=1):
    """Private function to prepare the sequence of operations for x-concatenated
    dynamical decoupling sequence

    Parameters
    ----------
    concatenation_sequence : int, optional
        Duration of the total sequence; defaults to 1

    Returns
    ------
    numpy.ndarray
        The offset values
    """

    if concatenation_sequence == 1:
        return np.array([1, 0, 1, 0])

    cumulated_operations = np.concatenate(
        (
            _concatenation_x(concatenation_sequence - 1),
            np.array([0]),
            _concatenation_x(concatenation_sequence - 1),
            np.array([0]),
        ),
        axis=0,
    )
    return cumulated_operations


def _concatenation_xy(concatenation_sequence=1):
    """Private function to prepare the sequence of operations for x-concatenated
    dynamical decoupling sequence

    Parameters
    ----------
    concatenation_sequence : int, optional
        Duration of the total sequence; defaults to 1

    Returns
    ------
    numpy.ndarray
        The offset values
    """

    if concatenation_sequence == 1:
        return np.array([1, -1, 1, -2, 1, -1, 1, -2])
    cumulations = np.concatenate(
        (_concatenation_xy(concatenation_sequence - 1), np.array([-1])), axis=0
    )
    cumulations = cumulations[0:-1]
    cumulations[-1] = -3
    cumulations = np.concatenate(
        (cumulations, _concatenation_xy(concatenation_sequence - 1), np.array([-2])),
        axis=0,
    )
    cumulations = cumulations[0:-2]
    cumulations = np.concatenate(
        (cumulations, _concatenation_xy(concatenation_sequence - 1), np.array([-1])),
        axis=0,
    )
    cumulations = cumulations[0:-1]
    cumulations[-1] = -3
    cumulations = np.concatenate(
        (cumulations, _concatenation_xy(concatenation_sequence - 1), np.array([-2])),
        axis=0,
    )
    if cumulations[-1] == -2 and cumulations[-2] == -2:
        cumulations = cumulations[0:-2]
    return cumulations
