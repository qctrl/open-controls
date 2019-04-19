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
===================
driven_controls.predefined
===================
"""

import numpy as np

from qctrlopencontrols.exceptions import ArgumentsValueError
from .driven_controls import DrivenControls

from qctrlopencontrols.globals import SQUARE, GAUSSIAN

from .constants import (
    PRIMITIVE, WIMPERIS_1, SOLOVAY_KITAEV_1,
    WALSH_AMPLITUDE_MODULATED_FILTER_1,
    COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE,
    COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_SOLOVAY_KITAEV,
    COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_WIMPERIS,
    SHORT_COMPOSITE_ROTATION_FOR_UNDOING_LENGTH_OVER_AND_UNDER_SHOOT,
    CORPSE_IN_SCROFULOUS_PULSE)

from .conversion import gaussian_max_rabi_rate_scale_down


def new_predefined_driven_control(
        driven_control_type=PRIMITIVE,
        **kwargs):
    """
    Create a new driven control

    Parameters
    ----------
    driven_control_type : string, optional
        Defaults to None. The name of the driven control type,
        supported options are:
        - 'primitive'
        - 'wimperis_1'
        - 'solovay_kitaev_1'
        - 'compensating_for_off_resonance_with_a_pulse_sequence'
        - 'compensating_for_off_resonance_with_a_pulse_sequence_with_wimperis'
        - 'compensating_for_off_resonance_with_a_pulse_sequence_with_solovay_kitaev'
        - 'walsh_amplitude_modulated_filter_1'
        - 'short_composite_rotation_for_undoing_length_over_and_under_shoot'
        - 'corpse_in_scrofulous'
    kwargs : dict, optional
        options to make the corresponding control type.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        Returns a driven control corresponding to the driven_control_type.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    # Forced to import here to avoid cyclic imports, need to review
    # Raise error if the input driven_control_type is not known
    if driven_control_type == PRIMITIVE:
        driven_control = new_primitive_control(**kwargs)
    elif driven_control_type == WIMPERIS_1:
        driven_control = new_wimperis_1_control(**kwargs)
    elif driven_control_type == SOLOVAY_KITAEV_1:
        driven_control = new_solovay_kitaev_1_control(**kwargs)
    elif driven_control_type == WALSH_AMPLITUDE_MODULATED_FILTER_1:
        driven_control = new_walsh_amplitude_modulated_filter_1_control(**kwargs)
    elif driven_control_type == COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE:
        driven_control = new_compensating_for_off_resonance_with_a_pulse_sequence_control(
            **kwargs)
    elif driven_control_type == COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_WIMPERIS:
        driven_control = \
            new_compensating_for_off_resonance_with_a_pulse_sequence_with_wimperis_control(
                **kwargs)
    elif driven_control_type == \
        COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_SOLOVAY_KITAEV:
        driven_control = \
            new_compensating_for_off_resonance_with_a_pulse_sequence_with_solovay_kitaev_control(
                **kwargs)
    elif driven_control_type == SHORT_COMPOSITE_ROTATION_FOR_UNDOING_LENGTH_OVER_AND_UNDER_SHOOT:
        driven_control = \
            new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control(**kwargs)
    elif driven_control_type == CORPSE_IN_SCROFULOUS_PULSE:
        driven_control = new_corpse_in_scrofulous_control(**kwargs)
    else:
        raise ArgumentsValueError(
            'Unknown predefined pulse type. See help(new_predefined_driven_control) to display all'
            + ' allowed inputs.',
            {'driven_control_type', driven_control_type})
    return driven_control

def _predefined_common_attributes(maximum_rabi_rate,
                                  rabi_rotation,
                                  shape,
                                  azimuthal_angle):
    """
    Adds some checks etc for all the predefined pulses

    Parameters
    ----------
    rabi_rotation : float
        The total polar angle to be performed by the pulse.
        Defined in polar coordinates.
    maximum_rabi_rate : float
        Defaults to 2.*np.pi
        The maximum rabi frequency for the pulse.
    shape : string
        The shape of the pulse.
    azimuthal_angle : float
        The azimuthal position of the pulse.

    Returns
    -------
    tuple
        Tuple of floats made of:
            (rabi_rate, rabi_rotation, azimuthal)

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """

    maximum_rabi_rate = float(maximum_rabi_rate)
    if maximum_rabi_rate <= 0:
        raise ArgumentsValueError(
            'Maximum rabi angular frequency should be greater than zero.',
            {'maximum_rabi_rate': maximum_rabi_rate})

    if shape == SQUARE:
        rabi_rate = maximum_rabi_rate
    elif shape == GAUSSIAN:
        rabi_rate = gaussian_max_rabi_rate_scale_down(maximum_rabi_rate)
    else:
        raise ArgumentsValueError(
            'The shape for a driven control must be either "{}" or "{}".'.format(SQUARE, GAUSSIAN),
            {'shape': shape})

    rabi_rotation = float(rabi_rotation)
    if rabi_rotation == 0:
        raise ArgumentsValueError(
            'The rabi rotation must be non zero.',
            {'rabi_rotation': rabi_rotation}
        )

    azimuthal_angle = float(azimuthal_angle)

    return (rabi_rate, rabi_rotation, azimuthal_angle)

def _get_transformed_rabi_rotation_wimperis(rabi_rotation):
    """
    Calculates the Rabi rotation angle as required by Wimperis 1 (BB1)
    and Solovay-Kitaev driven controls.

    Parameters
    ----------
    rabi_rotation : float
        Rotation angle of the operation

    Returns
    -------
    float
        The transformed angle as per definition for the Wimperis 1 (BB1) control

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    # Raise error if the polar angle is incorrect
    if rabi_rotation > 4 * np.pi:
        raise ArgumentsValueError(
            'The polar angle must be between -4 pi and 4 pi (inclusive).',
            {'rabi_rotation': rabi_rotation})
    return np.arccos(-rabi_rotation / (4 * np.pi))

def _derive_segments(angles, amplitude=2. * np.pi):
    """
    Derive the driven control segments from a set of rabi_rotations defined in terms of the
        spherical polar angles

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
    segments = [[amplitude * np.cos(phi), amplitude * np.sin(phi), 0., theta / amplitude]
                for (theta, phi) in angles]
    return segments


def new_primitive_control(
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    Primitive driven control.

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the driven control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the driven control.
    shape : str, optional
        Shape of the driven control.
    azimuthal_angle : float, optional
        The azimuthal position of the driven control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    segments = [[
        rabi_rate * np.cos(azimuthal_angle),
        rabi_rate * np.sin(azimuthal_angle),
        0.,
        rabi_rotation / rabi_rate], ]

    return DrivenControls(segments=segments, shape=shape, scheme=PRIMITIVE, **kwargs)


def new_wimperis_1_control(
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    Wimperis or BB1 control.

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the control.
    shape : str, optional
        Shape of the driven control.
    azimuthal_angle : float, optional
        The azimuthal position of the control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    phi_p = _get_transformed_rabi_rotation_wimperis(rabi_rotation)
    angles = np.array([
        [rabi_rotation, azimuthal_angle],
        [np.pi, phi_p + azimuthal_angle],
        [2 * np.pi, 3. * phi_p + azimuthal_angle],
        [np.pi, phi_p + azimuthal_angle]])

    segments = _derive_segments(angles, amplitude=rabi_rate)

    return DrivenControls(segments=segments, shape=shape, scheme=WIMPERIS_1, **kwargs)


def new_solovay_kitaev_1_control(
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    First-order Solovay-Kitaev control, also known as SK1

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the control.
    shape : str, optional
        Shape of the driven control.
    azimuthal_angle : float, optional
        The azimuthal position of the control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    phi_p = _get_transformed_rabi_rotation_wimperis(rabi_rotation)

    angles = np.array([
        [rabi_rotation, azimuthal_angle],
        [2 * np.pi, -phi_p + azimuthal_angle],
        [2 * np.pi, phi_p + azimuthal_angle]])

    segments = _derive_segments(angles, amplitude=rabi_rate)

    return DrivenControls(segments=segments, shape=shape, scheme=SOLOVAY_KITAEV_1, **kwargs)


def new_short_composite_rotation_for_undoing_length_over_and_under_shoot_control(  # pylint: disable=invalid-name
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    SCROFULOUS control to compensate for pulse length errors

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the control.
    shape : str, optional
        Shape of driven control.
    azimuthal_angle : float, optional
        The azimuthal position of the control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    # Create a lookup table for rabi rotation and phase angles, taken from the official paper.
    # Note: values in the paper are in degrees.
    def degrees_to_radians(angle_in_degrees):
        return angle_in_degrees / 180 * np.pi

    if np.isclose(rabi_rotation, np.pi):
        theta_1 = degrees_to_radians(180.)
        phi_1 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(- np.pi / 2 / theta_1)
    elif np.isclose(rabi_rotation, 0.5 * np.pi):
        theta_1 = degrees_to_radians(115.2)
        phi_1 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(- np.pi / 2 / theta_1)
    elif np.isclose(rabi_rotation, 0.25 * np.pi):
        theta_1 = degrees_to_radians(96.7)
        phi_1 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(- np.pi / 2 / theta_1)
    else:
        raise ArgumentsValueError(
            'rabi_rotation angle must be either pi, pi/2 or pi/4',
            {'rabi_rotation': rabi_rotation})

    theta_3 = theta_1
    phi_3 = phi_1
    theta_2 = np.pi

    angles = np.array([
        [theta_1, phi_1 + azimuthal_angle],
        [theta_2, phi_2 + azimuthal_angle],
        [theta_3, phi_3 + azimuthal_angle]])

    segments = _derive_segments(angles, amplitude=rabi_rate)

    return DrivenControls(
        segments=segments,
        shape=shape,
        scheme=SHORT_COMPOSITE_ROTATION_FOR_UNDOING_LENGTH_OVER_AND_UNDER_SHOOT,
        **kwargs)


def new_compensating_for_off_resonance_with_a_pulse_sequence_control(  # pylint: disable=invalid-name
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    Compensating for off resonance with a pulse sequence, often abbreviated as CORPSE.

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the control.
    shape : str, optional
        Shape of the driven control.
    azimuthal_angle : float, optional
        The azimuthal position of the control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    k = np.arcsin(np.sin(rabi_rotation / 2.) / 2.)
    angles = np.array([
        [2. * np.pi + rabi_rotation / 2. - k, azimuthal_angle],
        [2. * np.pi - 2. * k, np.pi + azimuthal_angle],
        [rabi_rotation / 2. - k, azimuthal_angle]])

    segments = _derive_segments(angles, amplitude=rabi_rate)

    return DrivenControls(
        segments=segments,
        shape=shape,
        scheme=COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE,
        **kwargs)


def new_compensating_for_off_resonance_with_a_pulse_sequence_with_wimperis_control(  # pylint: disable=invalid-name
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    Compensating for off resonance with a pulse sequence with an embedded
    Wimperis (or BB1) control, also known as CinBB.

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the control.
    shape : str, optional
        Shape of the driven control.
    azimuthal_angle : float, optional
        The azimuthal position of the control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    phi_p = _get_transformed_rabi_rotation_wimperis(rabi_rotation)
    k = np.arcsin(np.sin(rabi_rotation / 2.) / 2.)
    angles = np.array([
        [2. * np.pi + rabi_rotation / 2. - k, azimuthal_angle],
        [2. * np.pi - 2. * k, np.pi + azimuthal_angle],
        [rabi_rotation / 2. - k, azimuthal_angle],
        [np.pi, phi_p + azimuthal_angle],
        [2. * np.pi, 3 * phi_p + azimuthal_angle],
        [np.pi, phi_p + azimuthal_angle]])

    segments = _derive_segments(angles, amplitude=rabi_rate)

    return DrivenControls(
        segments=segments,
        shape=shape,
        scheme=COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_WIMPERIS,
        **kwargs)


def new_compensating_for_off_resonance_with_a_pulse_sequence_with_solovay_kitaev_control(  # pylint: disable=invalid-name
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    Compensating for off resonance with a pulse sequence with an
     embedded Solovay Kitaev (or SK1) control, also knowns as CinSK.

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the control.
    azimuthal_angle : float, optional
        The azimuthal position of the control.
    shape : str, optional
        Shape of the driven control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    phi_p = _get_transformed_rabi_rotation_wimperis(rabi_rotation)
    k = np.arcsin(np.sin(rabi_rotation / 2.) / 2.)
    angles = np.array([
        [2. * np.pi + rabi_rotation / 2. - k, azimuthal_angle],
        [2. * np.pi - 2. * k, np.pi + azimuthal_angle],
        [rabi_rotation / 2. - k, azimuthal_angle],
        [2. * np.pi, -phi_p + azimuthal_angle],
        [2. * np.pi, phi_p + azimuthal_angle]])

    segments = _derive_segments(angles, amplitude=rabi_rate)

    return DrivenControls(
        segments=segments,
        shape=shape,
        scheme=COMPENSATING_FOR_OFF_RESONANCE_WITH_A_PULSE_SEQUENCE_WITH_SOLOVAY_KITAEV,
        **kwargs)


def new_corpse_in_scrofulous_control(  # pylint: disable=invalid-name
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    CORPSE (Compensating for Off Resonance with a Pulse SEquence) embedded within a
    SCROFULOUS (Short Composite ROtation For Undoing Length Over and Under Shoot) control,
    also knowns as CinS.

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the control.
    azimuthal_angle : float, optional
        The azimuthal position of the control.
    shape : str, optional
        Shape of the driven control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    # Create a lookup table for rabi rotation and phase angles, taken from
    # the Cummings paper. Note: values in the paper are in degrees.
    def degrees_to_radians(angle_in_degrees):
        return angle_in_degrees / 180 * np.pi

    if np.isclose(rabi_rotation, np.pi):
        theta_1 = theta_3 = degrees_to_radians(180.)
        phi_1 = phi_3 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(- np.pi / 2 / theta_1)
    elif np.isclose(rabi_rotation, 0.5 * np.pi):
        theta_1 = theta_3 = degrees_to_radians(115.2)
        phi_1 = phi_3 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(- np.pi / 2 / theta_1)
    elif np.isclose(rabi_rotation, 0.25 * np.pi):
        theta_1 = theta_3 = degrees_to_radians(96.7)
        phi_1 = phi_3 = np.arccos(
            -np.pi * np.cos(theta_1) / 2 / theta_1 / np.sin(rabi_rotation / 2)
        )
        phi_2 = phi_1 - np.arccos(- np.pi / 2 / theta_1)
    else:
        raise ArgumentsValueError(
            'rabi_rotation angle must be either pi, pi/2 or pi/4',
            {'rabi_rotation': rabi_rotation})

    theta_2 = np.pi

    total_angles = []
    # Loop over all SCROFULOUS Rabi rotations (theta) and azimuthal angles (phi)
    # And make CORPSEs with those.
    for theta, phi in zip([theta_1, theta_2, theta_3], [phi_1, phi_2, phi_3]):
        k = np.arcsin(np.sin(theta / 2.) / 2.)
        angles = np.array([
            [2. * np.pi + theta / 2. - k, phi + azimuthal_angle],
            [2. * np.pi - 2. * k, np.pi + phi + azimuthal_angle],
            [theta / 2. - k, phi + azimuthal_angle]])
        total_angles.append(angles)

    total_angles = np.vstack(total_angles)

    segments = _derive_segments(total_angles, amplitude=rabi_rate)

    return DrivenControls(
        segments=segments,
        shape=shape,
        scheme=CORPSE_IN_SCROFULOUS_PULSE,
        **kwargs)


def new_walsh_amplitude_modulated_filter_1_control(  # pylint: disable=invalid-name
        rabi_rotation=None,
        azimuthal_angle=0.,
        maximum_rabi_rate=2. * np.pi,
        shape=SQUARE,
        **kwargs):
    """
    First order Walsh control with amplitude modulation.

    Parameters
    ----------
    rabi_rotation : float, optional
        The total rabi rotation to be performed by the control.
    maximum_rabi_rate : float, optional
        Defaults to 2.*np.pi
        The maximum rabi frequency for the control.
    azimuthal_angle : float, optional
        The azimuthal position of the control.
    shape : str, optional
        Shape of the driven control.
    kwargs : dict
        Other keywords required to make a qctrlopencontrols.DrivenControls.

    Returns
    -------
    qctrlopencontrols.DrivenControls
        The driven control.

    Raises
    ------
    ArgumentsValueError
        Raised when an argument is invalid.
    """
    (rabi_rate, rabi_rotation, azimuthal_angle) = _predefined_common_attributes(
        maximum_rabi_rate, rabi_rotation, shape, azimuthal_angle)

    if shape == SQUARE:
        if np.isclose(rabi_rotation, np.pi):
            theta_plus = np.pi
            theta_minus = np.pi / 2.
        elif np.isclose(rabi_rotation, 0.5 * np.pi):
            theta_plus = np.pi * (2.5 + 0.65667825) / 4.
            theta_minus = np.pi * (2.5 - 0.65667825) / 4.
        elif np.isclose(rabi_rotation, 0.25 * np.pi):
            theta_plus = np.pi * (2.25 + 0.36256159) / 4.
            theta_minus = np.pi * (2.25 - 0.36256159) / 4.
        else:
            raise ArgumentsValueError(
                'rabi_rotation angle must be either pi, pi/2 or pi/4',
                {'rabi_rotation': rabi_rotation})

            # Old on the fly general calc for square
            # Need to solve transcendental equation to get modulation depth factor
            # Have some precompiled solution, otherwise do it numerically
            # init_factor = 1.93296 - 0.220866 * (rabi_rotation / np.pi)
            # def factor_func(factor):
            #     return (
            #         ((1 - factor) * np.sin(rabi_rotation / 2.)
            #          + factor * np.sin((rabi_rotation * (factor - 1)) / (2. * (factor - 2.))))**2
            #     ) / ((factor - 1)**2)
            # modulation_depth_factor = newton(factor_func, init_factor)
            # assert 0. < modulation_depth_factor <= 2.
    else:  # shape == GAUSSIAN
        if np.isclose(rabi_rotation, np.pi):
            theta_plus = np.pi * (3 + 0.616016981956213) / 4
            theta_minus = np.pi * (3 - 0.616016981956213) / 4
        elif np.isclose(rabi_rotation, 0.5 * np.pi):
            theta_plus = np.pi * (2.5 + 0.4684796993336457) / 4
            theta_minus = np.pi * (2.5 - 0.4684796993336457) / 4
        elif np.isclose(rabi_rotation, 0.25 * np.pi):
            theta_plus = np.pi * (2.25 + 0.27723876925525176) / 4
            theta_minus = np.pi * (2.25 - 0.27723876925525176) / 4
        else:
            raise ArgumentsValueError(
                'rabi_rotation angle must be either pi, pi/2 or pi/4',
                {'rabi_rotation': rabi_rotation})

    rabi_rate_plus = rabi_rate
    time_segment = theta_plus / rabi_rate_plus
    rabi_rate_minus = theta_minus / time_segment

    segments = np.array([
        [rabi_rate_plus * np.cos(azimuthal_angle),
         rabi_rate_plus * np.sin(azimuthal_angle),
         0., time_segment],
        [rabi_rate_minus * np.cos(azimuthal_angle),
         rabi_rate_minus * np.sin(azimuthal_angle),
         0., time_segment],
        [rabi_rate_minus * np.cos(azimuthal_angle),
         rabi_rate_minus * np.sin(azimuthal_angle),
         0., time_segment],
        [rabi_rate_plus * np.cos(azimuthal_angle),
         rabi_rate_plus * np.sin(azimuthal_angle),
         0., time_segment]])

    return DrivenControls(
        segments=segments,
        shape=shape,
        scheme=WALSH_AMPLITUDE_MODULATED_FILTER_1,
        **kwargs)
