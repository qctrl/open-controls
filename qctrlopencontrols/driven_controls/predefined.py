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
Module for defining commonly used driven controls.
"""

from typing import (
    List,
    Optional,
)

import numpy as np

from ..utils import check_arguments
from .driven_control import DrivenControl


def _validate_rabi_parameters(rabi_rotation: float, maximum_rabi_rate: float) -> None:
    """
    Adds some checks etc for all the predefined pulses

    Parameters
    ----------
    rabi_rotation : float
        The total polar angle to be performed by the pulse.
        Defined in polar coordinates.
    maximum_rabi_rate : float
        The maximum rabi frequency for the pulse.
    """

    check_arguments(
        maximum_rabi_rate > 0,
        "Maximum rabi angular frequency should be greater than zero.",
        {"maximum_rabi_rate": maximum_rabi_rate},
    )

    check_arguments(
        rabi_rotation != 0,
        "The rabi rotation must be non-zero.",
        {"rabi_rotation": rabi_rotation},
    )


def _get_transformed_rabi_rotation_wimperis(rabi_rotation: float) -> float:
    """
    Calculates the Rabi rotation angle as required by Wimperis 1 (BB1)
    and Solovay-Kitaev driven controls.

    Parameters
    ----------
    rabi_rotation : float
        Rotation angle of the operation.

    Returns
    -------
    float
        The transformed angle as per definition for the Wimperis 1 (BB1) control.

    """

    # Raise error if the polar angle is incorrect
    check_arguments(
        -4 * np.pi <= rabi_rotation <= 4 * np.pi,
        "The polar angle must be between -4 pi and 4 pi (inclusive).",
        {"rabi_rotation": rabi_rotation},
    )
    return np.arccos(-rabi_rotation / (4 * np.pi))


def _derive_segments(
    angles: np.ndarray, amplitude: float = 2.0 * np.pi
) -> List[List[float]]:
    """
    Derive the driven control segments from a set of rabi_rotations defined in terms of the
    spherical polar angles.

    Parameters
    ----------
    angles : numpy.array
        angles is made of a list polar angle 2-lists formatted
        as [polar_angle, azimuthal_angle].
        All angles should be greater or equal to 0, and the polar_angles
        must be greater than zero.
    amplitude : float, optional
        Defaults to 1. The total amplitude of each segment in
        rad Hz.

    Returns
    -------
    list
        Segments for the driven control.

    """
    segments = [
        [amplitude * np.cos(phi), amplitude * np.sin(phi), 0.0, theta / amplitude]
        for (theta, phi) in angles
    ]
    return segments


def new_primitive_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a primitive (square) driven control.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    Notes
    -----
    A primitive driven control consists of a single control segment:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\theta/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    return DrivenControl(
        rabi_rates=[maximum_rabi_rate],
        azimuthal_angles=[azimuthal_angle],
        detunings=[0],
        durations=[rabi_rotation / maximum_rabi_rate],
        name=name,
    )


def new_bb1_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a BB1 (Wimperis) driven control.

    BB1 driven controls are robust to low-frequency noise sources that perturb the amplitude of
    the control field.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    Notes
    -----
    A BB1 driven control [#]_ consists of four control segments:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\theta/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
       :math:`\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\phi_*`, :math:`0`
       :math:`2\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+3\phi_*`,:math:`0`
       :math:`\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\phi_*`, :math:`0`

    where

    .. math::
        \phi_* = \cos^{-1} \left( -\frac{\theta}{4\pi} \right).

    References
    ----------
    .. [#] `S. Wimperis, Journal of Magnetic Resonance, Series A 109, 2 (1994).
        <https://doi.org/10.1006/jmra.1994.1159>`_
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    phi_p = _get_transformed_rabi_rotation_wimperis(rabi_rotation)
    rabi_rotations = [rabi_rotation, np.pi, 2 * np.pi, np.pi]

    rabi_rates = [maximum_rabi_rate] * 4
    azimuthal_angles = [
        azimuthal_angle,
        azimuthal_angle + phi_p,
        azimuthal_angle + 3 * phi_p,
        azimuthal_angle + phi_p,
    ]
    detunings = [0] * 4
    durations = [rabi_rotation / maximum_rabi_rate for rabi_rotation in rabi_rotations]

    return DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )


def new_sk1_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a first order Solovay-Kitaev (SK1) driven control.

    SK1 driven controls are robust to low-frequency noise sources that perturb the amplitude of
    the control field.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    Notes
    -----
    An SK1 driven control [#]_ [#]_ consists of three control segments:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\theta/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
       :math:`2\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi-\phi_*`, :math:`0`
       :math:`2\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\phi_*`, :math:`0`

    where

    .. math::
        \phi_* = \cos^{-1} \left( -\frac{\theta}{4\pi} \right).

    References
    ----------
    .. [#] `K. R. Brown, A. W. Harrow, and I. L. Chuang, Physical Review A 70, 052318 (2004).
        <https://doi.org/10.1103/PhysRevA.70.052318>`_
    .. [#] `K. R. Brown, A. W. Harrow, and I. L. Chuang, Physical Review A 72, 039905 (2005).
        <https://doi.org/10.1103/PhysRevA.72.039905>`_
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    phi_p = _get_transformed_rabi_rotation_wimperis(rabi_rotation)
    rabi_rotations = [rabi_rotation, 2 * np.pi, 2 * np.pi]

    rabi_rates = [maximum_rabi_rate] * 3
    azimuthal_angles = [
        azimuthal_angle,
        azimuthal_angle - phi_p,
        azimuthal_angle + phi_p,
    ]
    detunings = [0] * 3
    durations = [
        rabi_rotation_ / maximum_rabi_rate for rabi_rotation_ in rabi_rotations
    ]

    return DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )


def new_scrofulous_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a short composite rotation for undoing length over and under shoot (SCROFULOUS) driven
    control.

    SCROFULOUS driven controls are robust to low-frequency noise sources that perturb the amplitude
    of the control field.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control. Must be either
        :math:`\pi/4`, :math:`\pi/2`, or :math:`\pi`.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    Notes
    -----
    A SCROFULOUS driven control [#]_ consists of three control segments:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\theta_1/\Omega_{\rm max}`, :math:`\Omega_\rm{max}`, :math:`\phi+\phi_1`, :math:`0`
       :math:`\theta_2/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\phi_2`, :math:`0`
       :math:`\theta_3/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\phi_3`, :math:`0`

    where

    .. math::
        \theta_1 &= \theta_3 = \mathrm{sinc}^{-1} \left[\frac{2\cos (\theta/2)}{\pi}\right]

        \theta_2 &= \pi

        \phi_1 &= \phi_3 = \cos^{-1}\left[ \frac{-\pi\cos(\theta_1)}{2\theta_1\sin(\theta/2)}\right]

        \phi_2 &= \phi_1 - \cos^{-1} (-\pi/2\theta_1),

    and :math:`\mathrm{sinc}(x)=\sin(x)/x` is the unnormalized sinc function.

    References
    ----------
    .. [#] `H. K. Cummins, G. Llewellyn, and J. A. Jones, Physical Review A 67, 042308 (2003).
        <https://doi.org/10.1103/PhysRevA.67.042308>`_
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    # Create a lookup table for rabi rotation and phase angles, taken from the official paper.
    # Note: values in the paper are in degrees.
    def degrees_to_radians(angle_in_degrees):
        return angle_in_degrees / 180 * np.pi

    check_arguments(
        np.any(np.isclose(rabi_rotation, [np.pi, np.pi / 2, np.pi / 4])),
        "rabi_rotation angle must be either pi, pi/2 or pi/4",
        {"rabi_rotation": rabi_rotation},
    )

    if np.isclose(rabi_rotation, np.pi):
        theta_1 = degrees_to_radians(180.0)
        phi_1 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(-np.pi / 2 / theta_1)
    elif np.isclose(rabi_rotation, 0.5 * np.pi):
        theta_1 = degrees_to_radians(115.2)
        phi_1 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(-np.pi / 2 / theta_1)
    else:
        theta_1 = degrees_to_radians(96.7)
        phi_1 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(-np.pi / 2 / theta_1)

    theta_3 = theta_1
    phi_3 = phi_1
    theta_2 = np.pi

    rabi_rotations = [theta_1, theta_2, theta_3]

    rabi_rates = [maximum_rabi_rate] * 3
    azimuthal_angles = [
        azimuthal_angle + phi_1,
        azimuthal_angle + phi_2,
        azimuthal_angle + phi_3,
    ]
    detunings = [0] * 3
    durations = [
        rabi_rotation_ / maximum_rabi_rate for rabi_rotation_ in rabi_rotations
    ]

    return DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )


def new_corpse_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a compensating for off-resonance with a pulse sequence (CORPSE) driven control.

    CORPSE driven controls are robust to low-frequency dephasing noise.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    Notes
    -----
    A CORPSE driven control [#]_ [#]_ consists of three control segments:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\theta_1/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
       :math:`\theta_2/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\pi`, :math:`0`
       :math:`\theta_3/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`

    where

    .. math::
        \theta_1 &= 2\pi + \frac{\theta}{2} - \sin^{-1} \left[ \frac{\sin(\theta/2)}{2}\right]

        \theta_2 &= 2\pi - 2\sin^{-1} \left[ \frac{\sin(\theta/2)}{2}\right]

        \theta_3 &= \frac{\theta}{2} - \left[ \frac{\sin(\theta/2)}{2}\right].

    References
    ----------
    .. [#] `H. K. Cummins and J. A. Jones, New Journal of Physics 2 (2000).
        <https://doi.org/10.1088/1367-2630/2/1/006>`_
    .. [#] `H. K. Cummins, G. Llewellyn, and J. A. Jones, Physical Review A 67, 042308 (2003).
        <https://doi.org/10.1103/PhysRevA.67.042308>`_
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    k = np.arcsin(np.sin(rabi_rotation / 2.0) / 2.0)
    rabi_rotations = [
        rabi_rotation / 2.0 + 2 * np.pi - k,
        2 * np.pi - 2 * k,
        rabi_rotation / 2.0 - k,
    ]

    rabi_rates = [maximum_rabi_rate] * 3
    azimuthal_angles = [azimuthal_angle, azimuthal_angle + np.pi, azimuthal_angle]
    detunings = [0] * 3
    durations = [
        rabi_rotation_ / maximum_rabi_rate for rabi_rotation_ in rabi_rotations
    ]

    return DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )


def new_corpse_in_bb1_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a CORPSE concatenated within BB1 (CORPSE in BB1) driven control.

    CORPSE in BB1 driven controls are robust to both low-frequency noise sources that perturb the
    amplitude of the control field and low-frequency dephasing noise.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    See Also
    --------
    new_corpse_control, new_bb1_control

    Notes
    -----
    A CORPSE in BB1 driven control [#]_ [#]_ consists of a BB1 control with the first segment
    replaced by a CORPSE control, which yields six segments:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\theta_1/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
       :math:`\theta_2/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\pi`, :math:`0`
       :math:`\theta_3/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
       :math:`\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\phi_*`, :math:`0`
       :math:`2\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+3\phi_*`, :math:`0`
       :math:`\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\phi_*`, :math:`0`

    where

    .. math::
        \theta_1 &= 2\pi + \frac{\theta}{2} - \sin^{-1} \left[ \frac{\sin(\theta/2)}{2}\right]

        \theta_2 &= 2\pi - 2\sin^{-1} \left[ \frac{\sin(\theta/2)}{2}\right]

        \theta_3 &= \frac{\theta}{2} - \left[ \frac{\sin(\theta/2)}{2}\right]

        \phi_* &= \cos^{-1} \left( -\frac{\theta}{4\pi} \right).

    References
    ----------
    .. [#] `M. Bando, T. Ichikawa, Y Kondo, and M. Nakahara, Journal of the Physical Society of
        Japan 82, 1 (2012). <https://doi.org/10.7566/JPSJ.82.014004>`_
    .. [#] `C. Kabytayev, T. J. Green, K. Khodjasteh, M. J. Biercuk, L. Viola, and K. R. Brown,
        Physical Review A 90, 012316 (2014). <https://doi.org/10.1103/PhysRevA.90.012316>`_
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    phi_p = _get_transformed_rabi_rotation_wimperis(rabi_rotation)
    k = np.arcsin(np.sin(rabi_rotation / 2.0) / 2.0)

    rabi_rotations = [
        2 * np.pi + rabi_rotation / 2.0 - k,
        2 * np.pi - 2 * k,
        rabi_rotation / 2.0 - k,
        np.pi,
        2 * np.pi,
        np.pi,
    ]

    rabi_rates = [maximum_rabi_rate] * 6
    azimuthal_angles = [
        azimuthal_angle,
        azimuthal_angle + np.pi,
        azimuthal_angle,
        azimuthal_angle + phi_p,
        azimuthal_angle + 3 * phi_p,
        azimuthal_angle + phi_p,
    ]
    detunings = [0] * 6
    durations = [
        rabi_rotation_ / maximum_rabi_rate for rabi_rotation_ in rabi_rotations
    ]

    return DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )


def new_corpse_in_sk1_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a CORPSE concatenated within SK1 (CORPSE in SK1) driven control.

    CORPSE in SK1 driven controls are robust to both low-frequency noise sources that perturb the
    amplitude of the control field and low-frequency dephasing noise.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    See Also
    --------
    new_corpse_control, new_sk1_control

    Notes
    -----
    A CORPSE in SK1 driven control [#]_ [#]_ consists of an SK1 control with the first segment
    replaced by a CORPSE control, which yields five segments:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\theta_1/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
       :math:`\theta_2/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\pi`, :math:`0`
       :math:`\theta_3/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
       :math:`2\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi-\phi_*`, :math:`0`
       :math:`2\pi/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi+\phi_*`, :math:`0`

    where

    .. math::
        \theta_1 &= 2\pi + \frac{\theta}{2} - \sin^{-1} \left[ \frac{\sin(\theta/2)}{2}\right]

        \theta_2 &= 2\pi - 2\sin^{-1} \left[ \frac{\sin(\theta/2)}{2}\right]

        \theta_3 &= \frac{\theta}{2} - \left[ \frac{\sin(\theta/2)}{2}\right]

        \phi_* &= \cos^{-1} \left( -\frac{\theta}{4\pi} \right).

    References
    ----------
    .. [#] `M. Bando, T. Ichikawa, Y Kondo, and M. Nakahara, Journal of the Physical Society of
        Japan 82, 1 (2012). <https://doi.org/10.7566/JPSJ.82.014004>`_
    .. [#] `C. Kabytayev, T. J. Green, K. Khodjasteh, M. J. Biercuk, L. Viola, and K. R. Brown,
        Physical Review A 90, 012316 (2014). <https://doi.org/10.1103/PhysRevA.90.012316>`_
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    phi_p = _get_transformed_rabi_rotation_wimperis(rabi_rotation)
    k = np.arcsin(np.sin(rabi_rotation / 2.0) / 2.0)

    rabi_rotations = [
        2 * np.pi + rabi_rotation / 2.0 - k,
        2 * np.pi - 2 * k,
        rabi_rotation / 2.0 - k,
        2 * np.pi,
        2 * np.pi,
    ]

    rabi_rates = [maximum_rabi_rate] * 5
    azimuthal_angles = [
        azimuthal_angle,
        azimuthal_angle + np.pi,
        azimuthal_angle,
        azimuthal_angle - phi_p,
        azimuthal_angle + phi_p,
    ]
    detunings = [0] * 5
    durations = [
        rabi_rotation_ / maximum_rabi_rate for rabi_rotation_ in rabi_rotations
    ]

    return DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )


def new_corpse_in_scrofulous_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a CORPSE concatenated within SCROFULOUS (CORPSE in SCROFULOUS) driven control.

    CORPSE in SCROFULOUS driven controls are robust to both low-frequency noise sources that perturb
    the amplitude of the control field and low-frequency dephasing noise.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control. Must be either
        :math:`\pi/4`, :math:`\pi/2`, or :math:`\pi`.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    See Also
    --------
    new_corpse_control, new_scrofulous_control

    Notes
    -----
    A CORPSE in SCROFULOUS driven control [#]_ consists of a SCROFULOUS control with each segment
    replaced by a CORPSE control, which yields nine segments:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\Gamma^{\theta_1}_1/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_1`", :math:`0`
       :math:`\Gamma^{\theta_1}_2/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_1+\pi`", :math:`0`
       :math:`\Gamma^{\theta_1}_3/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_1`", :math:`0`
       :math:`\Gamma^{\theta_2}_1/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_2`", :math:`0`
       :math:`\Gamma^{\theta_2}_2/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_2+\pi`", :math:`0`
       :math:`\Gamma^{\theta_2}_3/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_2`", :math:`0`
       :math:`\Gamma^{\theta_3}_1/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_3`", :math:`0`
       :math:`\Gamma^{\theta_3}_2/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_3+\pi`", :math:`0`
       :math:`\Gamma^{\theta_3}_3/\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, "
       :math:`\phi+\phi_3`", :math:`0`

    where

    .. math::
        \theta_1 &= \theta_3 = \mathrm{sinc}^{-1} \left[\frac{2\cos (\theta/2)}{\pi}\right]

        \theta_2 &= \pi

        \phi_1 &= \phi_3 = \cos^{-1}\left[ \frac{-\pi\cos(\theta_1)}{2\theta_1\sin(\theta/2)}\right]

        \phi_2 &= \phi_1 - \cos^{-1} (-\pi/2\theta_1)

    (with :math:`\mathrm{sinc}(x)=\sin(x)/x` the unnormalized sinc function) are the SCROFULOUS
    angles, and

    .. math::
        \Gamma^{\theta'}_1 &= 2\pi + \frac{\theta'}{2}
            - \sin^{-1} \left[ \frac{\sin(\theta'/2)}{2}\right]

        \Gamma^{\theta'}_2 &= 2\pi - 2\sin^{-1} \left[ \frac{\sin(\theta'/2)}{2}\right]

        \Gamma^{\theta'}_3 &= \frac{\theta'}{2} - \left[ \frac{\sin(\theta'/2)}{2}\right]

    are the CORPSE angles corresponding to each SCROFULOUS angle
    :math:`\theta'\in\{\theta_1,\theta_2,\theta_3\}`.

    References
    ----------
    .. [#] `T. Ichikawa, M. Bando, Y. Kondo, and M. Nakahara, Physical Review A 84, 062311 (2011).
        <https://doi.org/10.1103/PhysRevA.84.062311>`_
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    check_arguments(
        np.any(np.isclose(rabi_rotation, [np.pi, np.pi / 2, np.pi / 4])),
        "rabi_rotation angle must be either pi, pi/2 or pi/4",
        {"rabi_rotation": rabi_rotation},
    )

    # Create a lookup table for rabi rotation and phase angles, taken from
    # the Cummins paper. Note: values in the paper are in degrees.
    def degrees_to_radians(angle_in_degrees):
        return angle_in_degrees / 180 * np.pi

    if np.isclose(rabi_rotation, np.pi):
        theta_1 = theta_3 = degrees_to_radians(180.0)
        phi_1 = phi_3 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(-np.pi / 2 / theta_1)
    elif np.isclose(rabi_rotation, 0.5 * np.pi):
        theta_1 = theta_3 = degrees_to_radians(115.2)
        phi_1 = phi_3 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(-np.pi / 2 / theta_1)
    else:
        theta_1 = theta_3 = degrees_to_radians(96.7)
        phi_1 = phi_3 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(-np.pi / 2 / theta_1)

    theta_2 = np.pi

    _total_angles = []
    # Loop over all SCROFULOUS Rabi rotations (theta) and azimuthal angles (phi)
    # And make CORPSEs with those.
    for theta, phi in zip([theta_1, theta_2, theta_3], [phi_1, phi_2, phi_3]):
        k = np.arcsin(np.sin(theta / 2.0) / 2.0)
        angles = np.array(
            [
                [2.0 * np.pi + theta / 2.0 - k, phi + azimuthal_angle],
                [2.0 * np.pi - 2.0 * k, np.pi + phi + azimuthal_angle],
                [theta / 2.0 - k, phi + azimuthal_angle],
            ]
        )
        _total_angles.append(angles)

    total_angles = np.vstack(_total_angles)
    rabi_rotations = total_angles[:, 0]

    rabi_rates = [maximum_rabi_rate] * 9
    azimuthal_angles = total_angles[:, 1]
    detunings = [0] * 9
    durations = [rabi_rotation / maximum_rabi_rate for rabi_rotation in rabi_rotations]

    return DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )


def new_wamf1_control(
    rabi_rotation: float,
    maximum_rabi_rate: float,
    azimuthal_angle: float = 0.0,
    name: Optional[str] = None,
) -> DrivenControl:
    r"""
    Creates a first-order Walsh amplitude-modulated filter (WAMF1) driven control.

    WAMF1 driven controls are robust to low-frequency dephasing noise.

    Parameters
    ----------
    rabi_rotation : float
        The total Rabi rotation :math:`\theta` to be performed by the driven control. Must be either
        :math:`\pi/4`, :math:`\pi/2`, or :math:`\pi`.
    maximum_rabi_rate : float
        The maximum Rabi frequency :math:`\Omega_{\rm max}` for the driven control.
    azimuthal_angle : float, optional
        The azimuthal angle :math:`\phi` for the rotation. Defaults to 0.
    name : str, optional
        An optional string to name the control. Defaults to ``None``.

    Returns
    -------
    DrivenControl
        The driven control :math:`\{(\delta t_n, \Omega_n, \phi_n, \Delta_n)\}`.

    Notes
    -----
    A WAMF1 [#]_ driven control consists of four control segments:

    .. csv-table::
       :header: :math:`\\delta t_n`, :math:`\\Omega_n`, :math:`\\phi_n` , :math:`\\Delta_n`

       :math:`\theta_+/4\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`
       :math:`\theta_+/4\Omega_{\rm max}`, :math:`\Omega_{\rm max}\theta_-/\theta_+`,"
       :math:`\phi`", :math:`0`
       :math:`\theta_+/4\Omega_{\rm max}`, :math:`\Omega_{\rm max}\theta_-/\theta_+`, "
       :math:`\phi`", :math:`0`
       :math:`\theta_+/4\Omega_{\rm max}`, :math:`\Omega_{\rm max}`, :math:`\phi`, :math:`0`

    where :math:`\theta_\pm = \theta+2\pi k_\theta\pm \delta_\theta`, and the integer
    :math:`k_\theta` and offset :math:`\delta_\theta` are optimized numerically in order to maximize
    the suppression of dephasing noise. Note that the optimal values depend only on the rotation
    angle :math:`\theta`.

    This implementation supports :math:`\theta\in\{\pi/4,\pi/2,\pi\}`.

    References
    ----------
    .. [#] `H. Ball and M. J. Biercuk, EPJ Quantum Technology 2, 11 (2015).
        <https://doi.org/10.1140/epjqt/s40507-015-0022-4>`_
    """

    _validate_rabi_parameters(
        rabi_rotation=rabi_rotation, maximum_rabi_rate=maximum_rabi_rate
    )

    check_arguments(
        np.any(np.isclose(rabi_rotation, [np.pi, np.pi / 2, np.pi / 4])),
        "rabi_rotation angle must be either pi, pi/2 or pi/4",
        {"rabi_rotation": rabi_rotation},
    )

    if np.isclose(rabi_rotation, np.pi):
        theta_plus = np.pi
        theta_minus = np.pi / 2.0
    elif np.isclose(rabi_rotation, 0.5 * np.pi):
        theta_plus = np.pi * (2.5 + 0.65667825) / 4.0
        theta_minus = np.pi * (2.5 - 0.65667825) / 4.0
    else:
        theta_plus = np.pi * (2.25 + 0.36256159) / 4.0
        theta_minus = np.pi * (2.25 - 0.36256159) / 4.0

    rabi_rotations = [theta_plus, theta_minus, theta_minus, theta_plus]
    segment_duration = theta_plus / maximum_rabi_rate

    rabi_rates = [rabi_rotation / segment_duration for rabi_rotation in rabi_rotations]
    azimuthal_angles = [azimuthal_angle] * 4
    detunings = [0] * 4
    durations = [segment_duration] * 4

    return DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )


def new_modulated_gaussian_control(
    maximum_rabi_rate: float,
    minimum_segment_duration: float,
    duration: float,
    modulation_frequency: float,
) -> DrivenControl:
    """
    Generate a Gaussian driven control sequence modulated by a sinusoidal signal at a specific
    frequency.

    The net effect of this control sequence is an identity gate.

    Parameters
    ----------
    maximum_rabi_rate: float
        Maximum Rabi rate of the system.

    minimum_segment_duration : float
        Minimum length of each segment in the control sequence.

    duration : float
        Total duration of the control sequence.

    modulation_frequency: float
        Frequency of the modulation sinusoidal signal.

    Returns
    -------
    DrivenControl
        A control sequence as an instance of DrivenControl.
    """

    check_arguments(
        maximum_rabi_rate > 0.0,
        "Maximum Rabi rate must be greater than zero.",
        {"maximum_rabi_rate": maximum_rabi_rate},
    )

    check_arguments(
        minimum_segment_duration > 0.0,
        "Minimum segment duration must be greater than zero.",
        {"minimum_segment_duration": minimum_segment_duration},
    )

    check_arguments(
        duration > minimum_segment_duration,
        "Total duration must be greater than minimum segment duration.",
        {"duration": duration, "minimum_segment_duration": minimum_segment_duration,},
    )

    # default spread of the gaussian shaped pulse as a fraction of its duration
    _pulse_width = 0.1

    # default mean of the gaussian shaped pulse as a fraction of its duration
    _pulse_mean = 0.5

    min_required_upper_bound = np.sqrt(2 * np.pi) / (_pulse_width * duration)
    check_arguments(
        maximum_rabi_rate >= min_required_upper_bound,
        "Maximum Rabi rate must be large enough to permit a 2Pi rotation.",
        {"maximum_rabi_rate": maximum_rabi_rate},
        extras={
            "minimum required value for upper_bound "
            "(sqrt(2pi)/(0.1*maximum_duration))": min_required_upper_bound
        },
    )

    # work out exact segment duration
    segment_count = int(np.ceil(duration / minimum_segment_duration))
    segment_duration = duration / segment_count
    segment_start_times = np.arange(segment_count) * segment_duration
    segment_midpoints = segment_start_times + segment_duration / 2

    # prepare a base gaussian shaped pulse
    gaussian_mean = _pulse_mean * duration
    gaussian_width = _pulse_width * duration
    base_gaussian_segments = np.exp(
        -0.5 * ((segment_midpoints - gaussian_mean) / gaussian_width) ** 2
    )

    if modulation_frequency != 0:
        # prepare the modulation signals. We use sinusoids that are zero at the center of the pulse,
        # which ensures the pulses are antisymmetric about the center of the pulse and thus effect
        # a net zero rotation.
        modulation_signals = np.sin(
            2.0 * np.pi * modulation_frequency * (segment_midpoints - duration / 2)
        )
        # modulate the base gaussian
        modulated_gaussian_segments = base_gaussian_segments * modulation_signals

        # maximum segment value
        pulse_segments_maximum = np.max(modulated_gaussian_segments)
        # normalize to maximum Rabi rate
        modulated_gaussian_segments = (
            maximum_rabi_rate * modulated_gaussian_segments / pulse_segments_maximum
        )
    else:
        # for the zero-frequency pulse, we need to produce the largest possible full rotation (i.e.
        # multiple of 2pi) while respecting the maximum Rabi rate. Note that if the maximum Rabi
        # rate does not permit even a single rotation (which could happen to a small degree due to
        # discretization issues) then we allow values to exceed the maximum Rabi rate.
        normalized_gaussian_segments = base_gaussian_segments / np.max(
            base_gaussian_segments
        )
        maximum_rotation_angle = (
            segment_duration * np.sum(normalized_gaussian_segments) * maximum_rabi_rate
        )
        maximum_full_rotation_angle = max(
            maximum_rotation_angle - maximum_rotation_angle % (2 * np.pi), 2 * np.pi
        )
        modulated_gaussian_segments = (
            normalized_gaussian_segments
            * maximum_rabi_rate
            * (maximum_full_rotation_angle / maximum_rotation_angle)
        )

    azimuthal_angles = [0 if v >= 0 else np.pi for v in modulated_gaussian_segments]

    return DrivenControl(
        rabi_rates=np.abs(modulated_gaussian_segments),
        azimuthal_angles=azimuthal_angles,
        detunings=np.zeros(segment_count),
        durations=np.array([segment_duration] * segment_count),
    )
