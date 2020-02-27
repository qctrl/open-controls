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
========================================================
dynamic_decoupling_sequences.dynamic_decoupling_sequence
========================================================
"""

import numpy as np

from qctrlopencontrols.base.utils import create_repr_from_attributes
from ..exceptions.exceptions import ArgumentsValueError
from ..globals import (
    QCTRL_EXPANDED, CSV, CYLINDRICAL)

from ..dynamic_decoupling_sequences import (UPPER_BOUND_OFFSETS, MATPLOTLIB)
from .driven_controls import convert_dds_to_driven_control


class DynamicDecouplingSequence(object):   #pylint: disable=too-few-public-methods
    """
    Create a dynamic decoupling sequence.
    Can be made of perfect operations, or realistic pulses.

    Parameters
    ----------
    duration : float
        Defaults to 1. The total time in seconds for the sequence.
    offsets : list
        Defaults to None.
        The times offsets in s for the center of pulses.
        If None, defaults to one operation at halfway [0.5].
    rabi_rotations : list
        Default to None.
        The rabi rotations at each time offset.
        If None, defaults to np.pi at each time offset.
    azimuthal_angles : list
        Default to None.
        The azimuthal angles at each time offset.
        If None, defaults to 0 at each time offset.
    detuning_rotations : list
        Default to None.
        The detuning rotations at each time offset.
        If None, defaults to 0 at each time offset.
    name : str
        Name of the sequence; Defaults to None

    Raises
    ------
    qctrlopencontrols.exceptions.ArgumentsValueError
        is raised if one of the inputs is invalid.
    """

    def __init__(self,
                 duration=1.,
                 offsets=None,
                 rabi_rotations=None,
                 azimuthal_angles=None,
                 detuning_rotations=None,
                 name=None
                 ):

        self.duration = duration
        if self.duration <= 0.:
            raise ArgumentsValueError(
                'Sequence duration must be above zero:',
                {'duration': self.duration})

        if offsets is None:
            offsets = [0.5]

        self.offsets = np.array(offsets, dtype=np.float)
        if self.offsets.shape[0] > UPPER_BOUND_OFFSETS: # pylint: disable=unsubscriptable-object
            raise ArgumentsValueError(
                'Number of offsets is above the allowed number of maximum offsets. ',
                {'number_of_offsets': self.offsets.shape[0],    # pylint: disable=unsubscriptable-object
                 'allowed_maximum_offsets': UPPER_BOUND_OFFSETS})

        if np.any(self.offsets < 0.) or np.any(self.offsets > self.duration):
            raise ArgumentsValueError(
                'Offsets for dynamic decoupling sequence must be between 0 and sequence '
                'duration (inclusive). ',
                {'offsets': offsets,
                 'duration': duration})

        if rabi_rotations is None:
            rabi_rotations = np.pi * np.ones((len(self.offsets),))

        if azimuthal_angles is None:
            azimuthal_angles = np.zeros((len(self.offsets),))

        if detuning_rotations is None:
            detuning_rotations = np.zeros((len(self.offsets),))

        self.rabi_rotations = np.array(rabi_rotations, dtype=np.float)
        self.azimuthal_angles = np.array(azimuthal_angles, dtype=np.float)
        self.detuning_rotations = np.array(detuning_rotations, dtype=np.float)

        if len(self.rabi_rotations) != self.number_of_offsets:
            raise ArgumentsValueError(
                'rabi rotations must have the same length as offsets. ',
                {'offsets': offsets,
                 'rabi_rotations': rabi_rotations})

        if len(self.azimuthal_angles) != self.number_of_offsets:
            raise ArgumentsValueError(
                'azimuthal angles must have the same length as offsets. ',
                {'offsets': offsets,
                 'azimuthal_angles': azimuthal_angles})

        if len(self.detuning_rotations) != self.number_of_offsets:
            raise ArgumentsValueError(
                'detuning rotations must have the same length as offsets. ',
                {'offsets': offsets,
                 'detuning_rotations': detuning_rotations,
                 'len(detuning_rotations)': len(self.detuning_rotations),
                 'number_of_offsets': self.number_of_offsets})

        self.name = name
        if self.name is not None:
            self.name = str(self.name)

    @property
    def number_of_offsets(self):
        """Returns the number of offsets

        Returns
        ------
        int
            The number of offsets in the dynamic decoupling sequence
        """

        return len(self.offsets)

    def export(self):
        """ Returns a dictionary formatted for plotting using the qctrl-visualizer package.

        Returns
        -------
        dict
            Dictionary with plot data that can be used by the plot_sequences
            method of the qctrl-visualizer package. It has keywords 'Rabi'
            and 'Detuning'.
        """

        plot_dictionary = {}

        plot_r = self.rabi_rotations
        plot_theta = self.azimuthal_angles
        plot_offsets = self.offsets
        plot_detunings = self.detuning_rotations

        plot_dictionary["Rabi"] = [{'rotation': r*np.exp(1.j*theta), 'offset': t}
            for r, theta, t in zip(plot_r, plot_theta, plot_offsets) ]

        plot_dictionary["Detuning"] = [{'rotation': v, 'offset': t}
            for v, t in zip(plot_detunings, plot_offsets) ]

        return plot_dictionary

    def __repr__(self):
        """Returns a string representation for the object. The returned string looks like a valid
        Python expression that could be used to recreate the object, including default arguments.

        Returns
        -------
        str
            String representation of the object including the values of the arguments.
        """

        attributes = [
            'duration',
            'offsets',
            'rabi_rotations',
            'azimuthal_angles',
            'detuning_rotations',
            'name']

        return create_repr_from_attributes(self, attributes)

    def __str__(self):
        """Prepares a friendly string format for a Dynamic Decoupling Sequence
        """

        dd_sequence_string = list()

        if self.name is not None:
            dd_sequence_string.append('{}:'.format(self.name))

        dd_sequence_string.append('Duration = {}'.format(self.duration))

        pretty_offset = [str(offset/self.duration) for offset in list(self.offsets)]
        pretty_offset = ','.join(pretty_offset)

        dd_sequence_string.append('Offsets = [{}] x {}'.format(pretty_offset, self.duration))

        pretty_rabi_rotations = [
            str(rabi_rotation/np.pi) for rabi_rotation in list(self.rabi_rotations)]
        pretty_rabi_rotations = ','.join(pretty_rabi_rotations)

        dd_sequence_string.append('Rabi Rotations = [{}] x pi'.format(pretty_rabi_rotations))

        pretty_azimuthal_angles = [
            str(azimuthal_angle/np.pi) for azimuthal_angle in list(self.azimuthal_angles)]
        pretty_azimuthal_angles = ','.join(pretty_azimuthal_angles)

        dd_sequence_string.append('Azimuthal Angles = [{}] x pi'.format(pretty_azimuthal_angles))

        pretty_detuning_rotations = [
            str(detuning_rotation/np.pi) for detuning_rotation in list(self.detuning_rotations)]
        pretty_detuning_rotations = ','.join(pretty_detuning_rotations)

        dd_sequence_string.append(
            'Detuning Rotations = [{}] x pi'.format(pretty_detuning_rotations))

        dd_sequence_string = '\n'.join(dd_sequence_string)

        return dd_sequence_string

    def export_to_file(self, filename=None,
                       file_format=QCTRL_EXPANDED,
                       file_type=CSV,
                       coordinates=CYLINDRICAL,
                       maximum_rabi_rate=2*np.pi,
                       maximum_detuning_rate=2*np.pi):
        """Prepares and saves the dynamic decoupling sequence in a file.

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
        maximum_rabi_rate : float, optional
            Maximum Rabi Rate; Defaults to :math:`2\\pi`
        maximum_detuning_rate : float, optional
            Maximum Detuning Rate; Defaults to :math:`2\\pi`

        References
        ----------
        `Q-CTRL Control Data Format
        <https://docs.q-ctrl.com/output-data-formats#q-ctrl-hardware>` _.

        Raises
        ------
        ArgumentsValueError
            Raised if some of the parameters are invalid.

        Notes
        -----
        The sequence is converted to a driven control using the maximum rabi and detuning
        rate. The driven control is then exported. This is done to facilitate a coherent
        integration with Q-CTRL BLACK OPAL's 1-Qubit workspace.
        """

        driven_control = convert_dds_to_driven_control(
            dynamic_decoupling_sequence=self,
            maximum_rabi_rate=maximum_rabi_rate,
            maximum_detuning_rate=maximum_detuning_rate,
            name=self.name)

        driven_control.export_to_file(filename=filename,
                                      file_format=file_format,
                                      file_type=file_type,
                                      coordinates=coordinates)


if __name__ == '__main__':
    pass
