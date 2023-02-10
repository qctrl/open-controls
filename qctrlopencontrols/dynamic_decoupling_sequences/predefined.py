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
Module for defining commonly used dynamical decoupling sequences.
"""

from __future__ import annotations

import numpy as np

from ..utils import check_arguments
from .dynamic_decoupling_sequence import DynamicDecouplingSequence


def _add_pre_post_rotations(
    duration: float,
    offsets: np.ndarray,
    rabi_rotations: np.ndarray,
    azimuthal_angles: np.ndarray,
    detuning_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Adds a pre and post X rotation at the start and end of the sequence.

    Note that with these two pre and post X rotations, the net effect of the DDS does not
    necessarily have to be an identity, but it will always be either an identity or Z pi rotation.
    For example, given a CPMG sequence of odd number Y pi rotations in the middle with the pre
    (pi/2) and post(-pi/2) X rotations, the net effect will be a Z gate.

    This function assumes that the sequences only have X, Y, and Z pi-pulses.
    An exception is thrown if that is not the case.

    Parameters
    ----------
    duration : float
        The duration of the sequence
    offsets : np.ndarray
        Offsets of the sequence.
    rabi_rotations : np.ndarray
        Rabi rotations at each of the offsets.
    azimuthal_angles : np.ndarray
        Azimuthal angles at each of the offsets.
    detuning_rotations : np.ndarray
        Detuning rotations at each of the offsets

    Returns
    -------
    tuple
        Containing the (offsets, rabi_rotations, azimuthal_angles, detuning_rotations)
        resulting after the addition of pi/2 pulses at the start and end of the sequence.
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
    check_arguments(
        len(offsets) == x_pi_pulses + y_pi_pulses + z_pi_pulses,
        "Sequence contains pulses that are not X, Y, or Z pi-pulses.",
        {
            "rabi_rotations": rabi_rotations,
            "azimuthal_angles": azimuthal_angles,
            "detuning_rotations": detuning_rotations,
        },
    )

    # parameters for pre-post pulses
    rabi_value = np.pi / 2
    detuning_value = 0.0
    initial_azimuthal = 0.0  # for pre-pulse
    final_azimuthal = 0.0  # for post-pulse

    # The sequence will preserve the state |0> is it has an even number
    # of X and Y pi-pulses
    preserves_10 = (x_pi_pulses + y_pi_pulses) % 2 == 0

    # The sequence will preserve the state |0>+|1> is it has an even number
    # of Y and Z pi-pulses
    preserves_11 = (y_pi_pulses + z_pi_pulses) % 2 == 0

    # the direction of the post rotation depends on the property of DDS.
    # if the net effect of the sequences is an identity gate or Y rotation, the post rotation
    # is chosen to be -pi/2 X pulse, otherwise use pi/2 X pulse, to ensure the net effect is an
    # identity or Z rotation.
    if (preserves_10 and preserves_11) or (not preserves_10 and not preserves_11):
        final_azimuthal = np.pi

    offsets = np.insert(offsets, [0, offsets.shape[0]], [0, duration])
    rabi_rotations = np.insert(
        rabi_rotations, [0, rabi_rotations.shape[0]], [rabi_value, rabi_value]
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


def new_ramsey_sequence(
    duration, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the Ramsey sequence.

    Parameters
    ----------
    duration : float
        Total duration of the sequence :math:`\tau` (in seconds).
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi / 2}` rotation
        is added at the start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The Ramsey sequence.

    Notes
    -----
    Technically, the Ramsey sequence [#]_ does not decouple the system from the environment.
    Nevertheless, it is a useful sequence for characterization and testing protocols
    and hence it is included. The sequence is parameterized by the duration :math:`\tau`
    and contains no offsets in between the start and the end time of the sequence.

    References
    ----------
    .. [#] `N. F. Ramsey, Physical Review 78, 695 (1950).
        <https://link.aps.org/doi/10.1103/PhysRev.78.695>`_
    """
    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )

    if pre_post_rotation:
        offsets = duration * np.array([0.0, 1.0])
        rabi_rotations = np.array([np.pi / 2, np.pi / 2])
        azimuthal_angles = np.array([0.0, np.pi])
        detuning_rotations = np.zeros((2,))
    else:
        offsets = np.array([])
        rabi_rotations = np.array([])
        azimuthal_angles = np.array([])
        detuning_rotations = np.array([])

    return DynamicDecouplingSequence(
        duration=duration,
        offsets=offsets,
        rabi_rotations=rabi_rotations,
        azimuthal_angles=azimuthal_angles,
        detuning_rotations=detuning_rotations,
        name=name,
    )


def new_spin_echo_sequence(
    duration, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the spin echo sequence.

    Parameters
    ----------
    duration : float
        Total duration of the sequence :math:`\tau` (in seconds).
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The spin echo sequence.

    Notes
    -----
    The spin echo sequence [#]_ is parameterized by duration :math:`\tau`. There is a single
    :math:`X_{\pi}` unitary operation at :math:`t_1 = \frac{\tau}{2}`.

    References
    ----------
    .. [#] `E. L. Hahn, Physical Review 80, 580 (1950).
        <https://link.aps.org/doi/10.1103/PhysRev.80.580>`_
    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )

    offsets = np.array([duration / 2.0])
    rabi_rotations = np.array([np.pi])
    azimuthal_angles = np.zeros(1)
    detuning_rotations = np.zeros(1)

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
        name=name,
    )


def new_carr_purcell_sequence(
    duration, offset_count, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the Carr-Purcell sequence.

    Parameters
    ----------
    duration : float
        Total duration of the sequence :math:`\tau` (in seconds).
    offset_count : int
        Number of offsets :math:`n`.
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The Carr-Purcell sequence.

    See Also
    --------
    new_cpmg_sequence

    Notes
    -----
    The Carr-Purcell sequence [#]_ is parameterized by the number of offsets :math:`n`
    and duration :math:`\tau`. The sequence is made up of a set of :math:`X_{\pi}`
    operations applied at

    .. math::
        t_i = \frac{\tau}{n} \left(i -  \frac{1}{2}\right) \;,

    where :math:`i = 1, \cdots, n`.

    References
    ----------
    .. [#] `H. Y. Carr and E. M. Purcell, Physical Review 94, 630 (1954).
        <https://link.aps.org/doi/10.1103/PhysRev.94.630>`_
    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )
    check_arguments(
        offset_count >= 1,
        "Number of offsets must be positive.",
        {"offset_count": offset_count},
    )

    # in case a float number is passed
    offset_count = int(offset_count)
    offsets = _carr_purcell_meiboom_gill_offsets(duration, offset_count)

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
        name=name,
    )


def new_cpmg_sequence(
    duration, offset_count, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the Carr-Purcell-Meiboom-Gill sequence.

    Parameters
    ----------
    duration : float
        Total duration of the sequence :math:`\tau` (in seconds).
    offset_count : int
        Number of offsets :math:`n`.
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The Carr-Purcell-Meiboom-Gill sequence.

    See Also
    --------
    new_carr_purcell_sequence

    Notes
    -----
    The Carr-Purcell-Meiboom-Gill sequence [#]_ has the same timing and number of offsets as the
    Carr-Purcell sequence. However, the intermediate :math:`\pi` rotations are applied along the
    :math:`Y` axis. That is, it consists of :math:`Y_{\pi}` operations applied at times

    .. math::
        t_i = \frac{\tau}{n} \left(i - \frac{1}{2}\right) \;,

    where :math:`i = 1, \cdots, n`.

    References
    ----------
    .. [#] `S. Meiboom and D. Gill, Review of Scientific Instruments 29:8, 688 (1958).
        <https://doi.org/10.1063/1.1716296>`_
    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )
    check_arguments(
        offset_count >= 1,
        "Number of offsets must be positive.",
        {"offset_count": offset_count},
    )

    # in case a float number is passed
    offset_count = int(offset_count)
    offsets = _carr_purcell_meiboom_gill_offsets(duration, offset_count)
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
        name=name,
    )


def new_uhrig_sequence(
    duration, offset_count, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the Uhrig sequence.

    Parameters
    ----------
    duration : float
        Total duration of the sequence :math:`\tau` (in seconds).
    offset_count : int
        Number of offsets :math:`n`.
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The Uhrig sequence.

    Notes
    -----
    The Uhrig sequence [#]_ is parameterized by duration :math:`\tau` and number of
    offsets :math:`n`. The sequence consists of :math:`Y_{\pi}` operations at offsets given by

    .. math::
        t_i = \tau \sin^2 \left( \frac{i\pi}{2(n+1)} \right) \;,

    where :math:`i = 1, \cdots, n`.

    References
    ----------
    .. [#] `G. S. Uhrig, Physical Review Letters 98, 100504 (2007).
        <https://link.aps.org/doi/10.1103/PhysRevLett.98.100504>`_
    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )
    check_arguments(
        offset_count >= 1,
        "Number of offsets must be positive.",
        {"offset_count": offset_count},
    )

    # in case a float number is passed
    offset_count = int(offset_count)
    offsets = _uhrig_single_axis_offsets(duration, offset_count)
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
        name=name,
    )


def new_periodic_sequence(
    duration, offset_count, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the periodic sequence.

    Parameters
    ----------
    duration : float
        Total duration of the sequence :math:`\tau` (in seconds).
    offset_count : int
        Number of offsets :math:`n`.
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The periodic sequence.

    Notes
    -----
    The periodic sequence [#]_ is parameterized by duration :math:`\tau` and number of
    offsets :math:`n`. The sequence consists of :math:`X_{\pi}` operations at offsets given by

    .. math::
        t_i = \frac{\tau}{n + 1} \;,

    where :math:`i = 1, \cdots, n`.

    References
    ----------
    .. [#] `L. Viola and E. Knill, Physical Review Letters 90, 037901 (2003).
        <https://link.aps.org/doi/10.1103/PhysRevLett.90.037901>`_
    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positve.",
        {"duration": duration},
    )
    check_arguments(
        offset_count >= 1,
        "Number of offsets must be positive.",
        {"offset_count": offset_count},
    )

    # in case a float number is passed
    offset_count = int(offset_count)

    spacing = 1.0 / (offset_count + 1)
    deltas = np.array([k * spacing for k in range(1, offset_count + 1)])
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
        name=name,
    )


def new_walsh_sequence(
    duration, paley_order, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the Walsh sequence.

    Parameters
    ----------
    duration : float
        Total duration of the sequence :math:`\tau` (in seconds).
    paley_order : int
        The paley order :math:`k` of the Walsh sequence.
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The Walsh sequence.

    Notes
    -----
    The Walsh sequence is defined by the switching function :math:`y(t)` given by a
    Walsh function. To define the Walsh sequence, we first introduce the Rademacher
    function [#]_, which is defined as

    .. math::
        R_j(x) := {\rm sgn}\left[\sin(2^j \pi x)\right] \;, \quad\; x \in [0, 1]\;, \; j \geq 0 \;.

    The :math:`j`-th Rademacher function :math:`R_j(x)` is thus a periodic square wave switching
    :math:`2^{j-1}` times between :math:`\pm 1` over the interval :math:`[0, 1]`. The Walsh
    function of Paley order :math:`k` is denoted :math:`{\rm PAL}_k(x)` and defined as

    .. math::
        {\rm PAL}_k(x) = \Pi_{j = 1}^m R_j(x)^{b_j} \;, \quad\; x \in [0, 1] \;.

    where :math:`(b_m, b_{m-1}, \cdots, b_1)` is the binary representation of :math:`k`.
    That is

    .. math::
        k = b_m 2^{m-1} + b_{m-1}2^{m-2} + \cdots + b_12^0 \;,

    where :math:`m = m(k)` indexes the most significant binary bit of :math:`k`.

    The :math:`k`-th order Walsh sequence [#]_ is then defined by

    .. math::
        y(t) = {\rm PAL}_k(t / \tau) \;

    with offset times :math:`\{t_j / \tau\}` defined at the switching times of the Walsh function.

    References
    ----------
    .. [#] `H. Rademacher, Math. Ann. 87, 112–138 (1922).
        <https://doi.org/10.1007/BF01458040>`_

    .. [#] `H. Ball and M. J Biercuk, EPJ Quantum Technol. 2, 11 (2015).
        <https://doi.org/10.1140/epjqt/s40507-015-0022-4>`_
    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )
    check_arguments(
        1 <= paley_order <= 2000,
        "Paley order must be between 1 and 2000.",
        {"paley_order": paley_order},
    )

    # in case a float number is passed
    paley_order = int(paley_order)

    hamming_weight = int(np.floor(np.log2(paley_order))) + 1

    samples = 2**hamming_weight

    relative_offset = np.arange(1.0 / (2 * samples), 1.0, 1.0 / samples)

    binary_string = np.binary_repr(paley_order)
    binary_order = [int(binary_string[i]) for i in range(hamming_weight)]
    walsh_array = np.ones(samples)
    for i in range(hamming_weight):
        walsh_array *= (
            np.sign(np.sin(2 ** (i + 1) * np.pi * relative_offset))
            ** binary_order[hamming_weight - 1 - i]
        )

    _walsh_relative_offsets = []
    for i in range(samples - 1):
        if walsh_array[i] != walsh_array[i + 1]:
            _walsh_relative_offsets.append((i + 1) * (1.0 / samples))
    walsh_relative_offsets = np.array(_walsh_relative_offsets, dtype=float)

    offsets = duration * walsh_relative_offsets
    rabi_rotations = np.full(offsets.shape, np.pi)
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
        name=name,
    )


def new_quadratic_sequence(
    duration, inner_offset_count, outer_offset_count, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the quadratic sequence.

    Parameters
    ----------
    duration : float
        The total duration of the sequence :math:`\tau` (in seconds).
    inner_offset_count : int
        Number of inner :math:`Z_{\pi}` pulses :math:`n_1`.
    outer_offset_count : int
        Number of outer :math:`X_{\pi}` pulses :math:`n_2`.
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The quadratic sequence.

    See Also
    --------
    new_uhrig_sequence

    Notes
    -----
    The quadratic sequence [#]_ is parameterized by duration :math:`\tau`, number of inner offsets
    :math:`n_1`, and number of outer offsets :math:`n_2`. The outer sequence consists of
    :math:`n_2` pulses of type :math:`X_{\pi}`, which partition the time-domain into :math:`n_2+1`
    sub-intervals on which inner sequences consisting of :math:`n_1` pulses of type
    :math:`Z_{\pi}` are nested. The total number of offsets is :math:`n = n_1 + n_2(n_1 + 1)`.

    The pulse times for outer sequence :math:`(X_{\pi}^1, \cdots, X_{\pi}^{n_2})` are defined
    according to the Uhrig sequence for :math:`t \in [0, \tau]`. The :math:`j`-th
    :math:`X_{\pi}` pulse, therefore has timing offset defined by

    .. math::
        t_x^j = \tau \sin^2 \left[ \frac{j \pi}{2(n_2 + 1)}  \right] \;,

    where :math:`j = 1, \cdots, n_2`. On each sub-interval defined by the outer sequence,
    an inner sequence :math:`(Z_{\pi}^1, \cdots, Z_{\pi}^{n_1})` is implemented. The pulse times
    for the inner sequences are also defined according to the Uhrig sequence. The :math:`k`-th
    pulse of the :math:`j`-th inner sequence has timing offset defined by

    .. math::
        t_z(k, j) = (t_x^j - t_x^{j - 1}) \sin^2 \left[ \frac{k \pi} {2 (n_1 + 1)} \right]
                    + t_{x}^{j - 1} \;,

    where :math:`k = 1, \cdots, n_1` and :math:`j = 1, \cdots, n_2 + 1`.

    References
    ----------
    .. [#] `J. R. West, B. H. Fong, and D. A. Lidar,
        Physical Review Letters 104, 130501 (2010).
        <https://doi.org/10.1103/PhysRevLett.104.130501>`_
    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )
    check_arguments(
        inner_offset_count >= 1,
        "Number of offsets of inner pulses must be positive.",
        {"inner_offset_count": inner_offset_count},
    )
    check_arguments(
        outer_offset_count >= 1,
        "Number of offsets of outer pulses must be positive.",
        {"outer_offset_count": outer_offset_count},
    )

    inner_offset_count = int(inner_offset_count)
    outer_offset_count = int(outer_offset_count)
    outer_offsets = _uhrig_single_axis_offsets(duration, outer_offset_count)
    outer_offsets = np.insert(outer_offsets, [0, len(outer_offsets)], [0, duration])

    inner_durations = np.diff(outer_offsets)

    # offsets include inner and outer offsets
    # the extra 1 dimension in columns is where we add the outer offset back
    offsets = np.zeros((len(inner_durations), inner_offset_count + 1))
    for inner_duration_idx, inner_duration in enumerate(inner_durations):
        inner_offset = (
            _uhrig_single_axis_offsets(inner_duration, inner_offset_count)
            + outer_offsets[inner_duration_idx]
        )
        offsets[inner_duration_idx, 0:inner_offset_count] = inner_offset
    offsets[:, -1] = outer_offsets[1:]

    rabi_rotations = np.zeros(offsets.shape)
    detuning_rotations = np.zeros(offsets.shape)

    rabi_rotations[0:outer_offset_count, -1] = np.pi
    detuning_rotations[0 : (outer_offset_count + 1), 0:inner_offset_count] = np.pi

    offsets = offsets.flatten()
    rabi_rotations = rabi_rotations.flatten()
    detuning_rotations = detuning_rotations.flatten()

    # remove the last entry corresponding to the duration
    offsets = offsets[:-1]
    rabi_rotations = rabi_rotations[:-1]
    detuning_rotations = detuning_rotations[:-1]

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
        name=name,
    )


def new_x_concatenated_sequence(
    duration, concatenation_order, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the :math:`X`-concatenated sequence.

    Parameters
    ----------
    duration : float
        The total duration of the sequence :math:`\tau` (in seconds).
    concatenation_order : int
        The number of concatenation of base sequence.
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The :math:`X`-concatenated sequence.

    See Also
    --------
    new_xy_concatenated_sequence

    Notes
    -----
    The :math:`X`-concatenated sequence [#]_ is constructed by recursively concatenating
    control sequence structures. It's parameterized by the concatenation order :math:`l` and
    the duration of the total sequence :math:`\tau`. Let the :math:`l`-th order of concatenation
    be denoted as :math:`C_l(\tau)`. In this scheme, zeroth order concatenation of duration
    :math:`\tau` is defined as free evolution over a period of :math:`\tau`. Using the notation
    :math:`{\mathcal 1}(\tau)` to represent free evolution over duration :math:`\tau`, the
    the base sequence is:

    .. math::
        C_0(\tau) = {\mathcal 1}(\tau) \;.

    The :math:`l`-th order :math:`X`-concatenated sequence can be recursively defined as

    .. math::
        C_l(\tau) = C_{l - 1}(\tau / 2) X_{\pi} C_{l - 1}(\tau / 2) X_{\pi} \;.

    References
    ----------
    .. [#] `K. Khodjasteh and D. A. Lidar, Physical Review Letters 95, 180501 (2005).
        <https://doi.org/10.1103/PhysRevLett.95.180501>`_
    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )
    check_arguments(
        concatenation_order >= 1,
        "Concatenation order must be positive.",
        {"concatenation_order": concatenation_order},
    )

    concatenation_order = int(concatenation_order)
    unit_spacing = duration / (2**concatenation_order)
    cumulations = _concatenation_x(concatenation_order)

    pos_cum = cumulations * unit_spacing
    pos_cum_sum = np.cumsum(pos_cum)

    values, counts = np.unique(pos_cum_sum, return_counts=True)

    offsets = np.array(
        [value for value, count in zip(values, counts) if count % 2 == 0]
    )

    if concatenation_order % 2 == 1:
        offsets = offsets[:-1]

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
        name=name,
    )


def new_xy_concatenated_sequence(
    duration, concatenation_order, pre_post_rotation=False, name=None
) -> DynamicDecouplingSequence:
    r"""
    Creates the :math:`XY`-concatenated sequence.

    Parameters
    ----------
    duration : float
        The total duration of the sequence :math:`\tau` (in seconds).
    concatenation_order : int
        The number of concatenation of base sequence :math:`l`.
    pre_post_rotation : bool, optional
        If ``True``, a :math:`X_{\pi/2}` rotation is added at the
        start and end of the sequence. Defaults to ``False``.
    name : string, optional
        Name of the sequence. Defaults to ``None``.

    Returns
    -------
    DynamicDecouplingSequence
        The :math:`XY`-concatenated sequence.

    See Also
    --------
    new_x_concatenated_sequence

    Notes
    -----
    The :math:`XY`-concatenated sequence [#]_ is constructed by recursively concatenating
    control sequence structures. It's parameterized by the concatenation order :math:`l` and
    the duration of the total sequence :math:`\tau`. Let the :math:`l`-th order of concatenation
    be denoted as :math:`C_l(\tau)`. In this scheme, zeroth order concatenation of duration
    :math:`\tau` is defined as free evolution over a period of :math:`\tau`. Using the notation
    :math:`{\mathcal 1}(\tau)` to represent free evolution over duration :math:`\tau`, the
    the base sequence is:

    .. math::
        C_0(\tau) = {\mathcal 1}(\tau) \;.

    The :math:`l`-th order :math:`XY`-concatenated sequence can be recursively defined as

    .. math::
        C_l(\tau) = C_{l - 1}(\tau / 4) X_{\pi} C_{l - 1}(\tau / 4) Y_{\pi}
                    C_{l - 1}(\tau / 4) X_{\pi} C_{l - 1}(\tau / 4) Y_{\pi} \;.

    References
    ----------
    .. [#] `K. Khodjasteh and D. A. Lidar, Physical Review Letters 95, 180501 (2005).
        <https://doi.org/10.1103/PhysRevLett.95.180501>`_

    """

    check_arguments(
        duration > 0,
        "Sequence duration must be positive.",
        {"duration": duration},
    )
    check_arguments(
        concatenation_order >= 1,
        "Concatenation order must be positive.",
        {"concatenation_order": concatenation_order},
    )

    concatenation_order = int(concatenation_order)

    unit_spacing = duration / (2 ** (concatenation_order * 2))
    cumulations = _concatenation_xy(concatenation_order)

    rabi_operations = cumulations[cumulations != -2]
    rabi_operations = rabi_operations[rabi_operations != -3]
    rabi_positions = np.zeros(rabi_operations.shape)
    rabi_positions[rabi_operations != -1] = 1
    rabi_positions = rabi_positions * unit_spacing
    rabi_positions = np.cumsum(rabi_positions)

    values, counts = np.unique(rabi_positions, return_counts=True)
    rabi_offsets = [value for value, count in zip(values, counts) if count % 2 == 0]

    azimuthal_operations = cumulations[cumulations != -1]
    azimuthal_operations = azimuthal_operations[azimuthal_operations != -3]
    azimuthal_positions = np.zeros(azimuthal_operations.shape)
    azimuthal_positions[azimuthal_operations != -2] = 1
    azimuthal_positions = azimuthal_positions * unit_spacing
    azimuthal_positions = np.cumsum(azimuthal_positions)

    values, counts = np.unique(azimuthal_positions, return_counts=True)
    azimuthal_offsets = [
        value for value, count in zip(values, counts) if count % 2 == 0
    ]

    detuning_operations = cumulations[cumulations != -2]
    detuning_operations = detuning_operations[detuning_operations != -1]
    detuning_positions = np.zeros(detuning_operations.shape)
    detuning_positions[detuning_operations != -3] = 1
    detuning_positions = detuning_positions * unit_spacing
    detuning_positions = np.cumsum(detuning_positions)

    values, counts = np.unique(detuning_positions, return_counts=True)
    detuning_offsets = [value for value, count in zip(values, counts) if count % 2 == 0]

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
        name=name,
    )


def _carr_purcell_meiboom_gill_offsets(
    duration: float, offset_count: int
) -> np.ndarray:
    """
    Calculates offset values for Carr-Purcell_Meiboom-Gill sequence.

    Parameters
    ----------
    duration : float
        Duration of the total sequence.
    offset_count : int
        The number of offsets

    Returns
    -------
    np.ndarray
        The offset values.
    """

    spacing = 1.0 / offset_count
    start = spacing * 0.5

    # prepare the offsets for delta comb
    deltas = spacing * np.arange(offset_count)
    deltas += start
    offsets = deltas * duration

    return offsets


def _uhrig_single_axis_offsets(duration: float, offset_count: int) -> np.ndarray:
    """
    Calculates oOffset values for Uhrig Single Axis Sequence.

    Parameters
    ----------
    duration : float
        Duration of the total sequence.
    offset_count : int
        The number of offsets.

    Returns
    -------
    np.ndarray
        The offset values.
    """

    # prepare the offsets for delta comb
    constant = 1.0 / (2 * offset_count + 2)
    deltas = np.array(
        [(np.sin(np.pi * k * constant)) ** 2 for k in range(1, offset_count + 1)]
    )
    offsets = duration * deltas

    return offsets


def _concatenation_x(concatenation_sequence: int) -> np.ndarray:
    """
    Prepares the sequence of operations for x-concatenated
    dynamical decoupling sequence.

    Parameters
    ----------
    concatenation_sequence : int
        Duration of the total sequence.

    Returns
    -------
    np.ndarray
        The offset values.
    """

    if concatenation_sequence == 1:
        return np.array([1, 0, 1, 0])

    return np.concatenate(
        (
            _concatenation_x(concatenation_sequence - 1),
            np.array([0]),
            _concatenation_x(concatenation_sequence - 1),
            np.array([0]),
        ),
        axis=0,
    )


def _concatenation_xy(concatenation_sequence) -> np.ndarray:
    """
    Prepares the sequence of operations for x-concatenated
    dynamical decoupling sequence.

    Parameters
    ----------
    concatenation_sequence : int
        Duration of the total sequence.

    Returns
    -------
    np.ndarray
        The offset values.
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
