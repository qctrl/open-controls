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
Tests for driven controls.
"""


import os

import numpy as np
import pytest

from qctrlopencontrols import DrivenControl
from qctrlopencontrols.exceptions import ArgumentsValueError


def _remove_file(filename):
    """
    Removes the file after test done.
    """

    if os.path.exists(filename):
        os.remove(filename)
    else:
        raise IOError(f"Could not find file {filename}")


def test_driven_controls():

    """
    Tests the construction of driven controls.
    """
    _rabi_rates = np.array([np.pi, np.pi, 0])
    _azimuthal_angles = np.array([np.pi / 2, 0, -np.pi])
    _detunings = np.array([0, 0, 0])
    _durations = np.array([1, 2, 3])

    _name = "driven_control"

    driven_control = DrivenControl(
        rabi_rates=_rabi_rates,
        azimuthal_angles=_azimuthal_angles,
        detunings=_detunings,
        durations=_durations,
        name=_name,
    )

    assert np.allclose(driven_control.rabi_rates, _rabi_rates)
    assert np.allclose(driven_control.durations, _durations)
    assert np.allclose(driven_control.detunings, _detunings)
    assert np.allclose(driven_control.azimuthal_angles, _azimuthal_angles)
    assert driven_control.name == _name


def test_driven_control_default_values():
    """
    Tests driven control with default values and invalid input.
    """
    _rabi_rates = np.array([np.pi, np.pi, 0])
    _azimuthal_angles = np.array([np.pi / 2, 0, -np.pi])
    _detunings = np.array([0, 0, 0])
    _durations = np.array([1, 2, 3])

    _name = "driven_control"

    driven_control = DrivenControl(
        rabi_rates=None,
        azimuthal_angles=_azimuthal_angles,
        detunings=_detunings,
        durations=_durations,
        name=_name,
    )

    assert np.allclose(driven_control.rabi_rates, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(driven_control.durations, _durations)
    assert np.allclose(driven_control.detunings, _detunings)
    assert np.allclose(driven_control.azimuthal_angles, _azimuthal_angles)

    driven_control = DrivenControl(
        rabi_rates=_rabi_rates,
        azimuthal_angles=None,
        detunings=_detunings,
        durations=_durations,
        name=_name,
    )

    assert np.allclose(driven_control.rabi_rates, _rabi_rates)
    assert np.allclose(driven_control.durations, _durations)
    assert np.allclose(driven_control.detunings, _detunings)
    assert np.allclose(driven_control.azimuthal_angles, np.array([0.0, 0.0, 0.0]))

    driven_control = DrivenControl(
        rabi_rates=_rabi_rates,
        azimuthal_angles=_azimuthal_angles,
        detunings=None,
        durations=_durations,
        name=_name,
    )

    assert np.allclose(driven_control.rabi_rates, _rabi_rates)
    assert np.allclose(driven_control.durations, _durations)
    assert np.allclose(driven_control.detunings, np.array([0.0, 0.0, 0.0]))
    assert np.allclose(driven_control.azimuthal_angles, _azimuthal_angles)

    driven_control = DrivenControl(durations=np.array([1]))
    assert np.allclose(driven_control.rabi_rates, np.array([0.0]))
    assert np.allclose(driven_control.durations, np.array([1.0]))
    assert np.allclose(driven_control.detunings, np.array([0.0]))
    assert np.allclose(driven_control.azimuthal_angles, np.array([0.0]))

    with pytest.raises(ArgumentsValueError):
        _ = DrivenControl(durations=np.array([1]), rabi_rates=np.array([-1]))

    with pytest.raises(ArgumentsValueError):
        _ = DrivenControl(durations=np.array([0]))

    with pytest.raises(ArgumentsValueError):
        _ = DrivenControl(
            durations=np.array([1]),
            rabi_rates=np.array([1, 2]),
            azimuthal_angles=np.array([1, 2, 3]),
        )


def test_control_directions():
    """
    Tests if the directions method works properly.
    """
    rabi_rates = np.array([1, 0, 1, 0])
    azimuthal_angles = np.array([0, 0, np.pi / 2, 0])
    detunings = np.array([0, 0, 0, 1])
    durations = np.array([1, 1, 1, 1])

    name = "driven_control"

    expected_directions = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])

    driven_control = DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )

    assert np.allclose(driven_control.directions, expected_directions)


def test_control_directions_with_small_amplitudes():
    """
    Tests if the directions method works with very small amplitudes.
    """
    rabi_rates = np.array([1e-100, 0.0, 1e-100, 0.0])
    azimuthal_angles = np.array([0, 0, np.pi / 2, 0])
    detunings = np.array([0, 0, 0, 1e-100])
    durations = np.array([1, 1, 1, 1])

    name = "driven_control"

    expected_directions = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])

    driven_control = DrivenControl(
        rabi_rates=rabi_rates,
        azimuthal_angles=azimuthal_angles,
        detunings=detunings,
        durations=durations,
        name=name,
    )

    assert np.allclose(driven_control.directions, expected_directions)


def test_control_export():

    """
    Tests exporting the control to a file.
    """
    _rabi_rates = [5 * np.pi, 4 * np.pi, 3 * np.pi]
    _azimuthal_angles = [np.pi / 4, np.pi / 3, 0]
    _detunings = [0, 0, np.pi]
    _durations = [2, 2, 1]

    _name = "driven_control"

    driven_control = DrivenControl(
        rabi_rates=np.asarray(_rabi_rates),
        azimuthal_angles=np.asarray(_azimuthal_angles),
        detunings=np.asarray(_detunings),
        durations=np.asarray(_durations),
        name=_name,
    )

    _filename = "driven_control_qctrl_cylindrical.csv"
    driven_control.export_to_file(
        filename=_filename,
        file_format="Q-CTRL expanded",
        file_type="CSV",
        coordinates="cylindrical",
    )

    _filename = "driven_control_qctrl_cartesian.csv"
    driven_control.export_to_file(
        filename=_filename,
        file_format="Q-CTRL expanded",
        file_type="CSV",
        coordinates="cartesian",
    )

    _filename = "driven_control_qctrl_cylindrical.json"
    driven_control.export_to_file(
        filename=_filename,
        file_format="Q-CTRL expanded",
        file_type="JSON",
        coordinates="cylindrical",
    )

    _filename = "driven_control_qctrl_cartesian.json"
    driven_control.export_to_file(
        filename=_filename,
        file_format="Q-CTRL expanded",
        file_type="JSON",
        coordinates="cartesian",
    )

    _remove_file("driven_control_qctrl_cylindrical.csv")
    _remove_file("driven_control_qctrl_cartesian.csv")
    _remove_file("driven_control_qctrl_cylindrical.json")
    _remove_file("driven_control_qctrl_cartesian.json")


def test_plot_data():
    """
    Test the plot data produced for a driven control.
    """
    _rabi_rates = [np.pi, 2 * np.pi, np.pi]
    _azimuthal_angles = [0, np.pi / 2, -np.pi / 2]
    _detunings = [0, 1, 0]
    _durations = [1, 1.25, 1.5]

    driven_control = DrivenControl(
        rabi_rates=np.asarray(_rabi_rates),
        azimuthal_angles=np.asarray(_azimuthal_angles),
        detunings=np.asarray(_detunings),
        durations=np.asarray(_durations),
    )

    x_amplitude = [np.pi, 0.0, 0.0]
    y_amplitude = [0.0, 2 * np.pi, -np.pi]

    plot_data = driven_control.export(
        dimensionless_rabi_rate=False, coordinates="cartesian"
    )

    assert np.allclose(
        [point["duration"] for point in plot_data["X amplitude"]], _durations
    )
    assert np.allclose(
        [point["duration"] for point in plot_data["Y amplitude"]], _durations
    )
    assert np.allclose(
        [point["duration"] for point in plot_data["Detuning"]], _durations
    )

    assert np.allclose(
        [point["value"] for point in plot_data["X amplitude"]], x_amplitude
    )
    assert np.allclose(
        [point["value"] for point in plot_data["Y amplitude"]], y_amplitude
    )
    assert np.allclose([point["value"] for point in plot_data["Detuning"]], _detunings)


def test_pretty_print():

    """
    Tests pretty output of driven control.
    """

    _maximum_rabi_rate = 2 * np.pi
    _maximum_detuning = 1.0
    _rabi_rates = [np.pi, 2 * np.pi, np.pi]
    _azimuthal_angles = [0, np.pi / 2, -np.pi / 2]
    _detunings = [0, 1, 0]
    _durations = [1.0, 1.0, 1.0]

    driven_control = DrivenControl(
        rabi_rates=np.asarray(_rabi_rates),
        azimuthal_angles=np.asarray(_azimuthal_angles),
        detunings=np.asarray(_detunings),
        durations=np.asarray(_durations),
    )

    _pretty_rabi_rates = ",".join(
        [str(_rabi_rate / _maximum_rabi_rate) for _rabi_rate in _rabi_rates]
    )
    _pretty_azimuthal_angles = ",".join(
        [str(azimuthal_angle / np.pi) for azimuthal_angle in _azimuthal_angles]
    )
    _pretty_detunings = ",".join(
        [str(detuning / _maximum_detuning) for detuning in _detunings]
    )
    _pretty_durations = ",".join([str(duration / 3.0) for duration in _durations])

    _pretty_string = []
    _pretty_string.append(f"Rabi Rates = [{_pretty_rabi_rates}] × {_maximum_rabi_rate}")
    _pretty_string.append(f"Azimuthal Angles = [{_pretty_azimuthal_angles}] × pi")
    _pretty_string.append(f"Detunings = [{_pretty_detunings}] × {_maximum_detuning}")
    _pretty_string.append(f"Durations = [{_pretty_durations}] × 3.0")

    expected_string = "\n".join(_pretty_string)

    assert str(driven_control) == expected_string

    _maximum_rabi_rate = 0.0
    _maximum_detuning = 1.0
    _rabi_rates = [0.0, 0.0, 0.0]
    _azimuthal_angles = [0, np.pi / 2, -np.pi / 2]
    _detunings = [0, 1, 0]
    _durations = [1.0, 1.0, 1.0]

    driven_control = DrivenControl(
        rabi_rates=np.asarray(_rabi_rates),
        azimuthal_angles=np.asarray(_azimuthal_angles),
        detunings=np.asarray(_detunings),
        durations=np.asarray(_durations),
    )

    _pretty_rabi_rates = ",".join(["0", "0", "0"])
    _pretty_azimuthal_angles = ",".join(
        [str(azimuthal_angle / np.pi) for azimuthal_angle in _azimuthal_angles]
    )
    _pretty_detunings = ",".join(
        [str(detuning / _maximum_detuning) for detuning in _detunings]
    )
    _pretty_durations = ",".join([str(duration / 3.0) for duration in _durations])

    _pretty_string = []
    _pretty_string.append(f"Rabi Rates = [{_pretty_rabi_rates}] × {_maximum_rabi_rate}")
    _pretty_string.append(f"Azimuthal Angles = [{_pretty_azimuthal_angles}] × pi")
    _pretty_string.append(f"Detunings = [{_pretty_detunings}] × {_maximum_detuning}")
    _pretty_string.append(f"Durations = [{_pretty_durations}] × 3.0")

    expected_string = "\n".join(_pretty_string)

    assert str(driven_control) == expected_string

    _maximum_rabi_rate = 2 * np.pi
    _maximum_detuning = 0.0
    _rabi_rates = [np.pi, 2 * np.pi, np.pi]
    _azimuthal_angles = [0, np.pi / 2, -np.pi / 2]
    _detunings = [0, 0, 0]
    _durations = [1.0, 1.0, 1.0]

    driven_control = DrivenControl(
        rabi_rates=np.asarray(_rabi_rates),
        azimuthal_angles=np.asarray(_azimuthal_angles),
        detunings=np.asarray(_detunings),
        durations=np.asarray(_durations),
    )

    _pretty_rabi_rates = ",".join(
        [str(_rabi_rate / _maximum_rabi_rate) for _rabi_rate in _rabi_rates]
    )
    _pretty_azimuthal_angles = ",".join(
        [str(azimuthal_angle / np.pi) for azimuthal_angle in _azimuthal_angles]
    )
    _pretty_detunings = ",".join(["0", "0", "0"])
    _pretty_durations = ",".join([str(duration / 3.0) for duration in _durations])

    _pretty_string = []
    _pretty_string.append(f"Rabi Rates = [{_pretty_rabi_rates}] × {_maximum_rabi_rate}")
    _pretty_string.append(f"Azimuthal Angles = [{_pretty_azimuthal_angles}] × pi")
    _pretty_string.append(f"Detunings = [{_pretty_detunings}] × {_maximum_detuning}")
    _pretty_string.append(f"Durations = [{_pretty_durations}] × 3.0")

    expected_string = "\n".join(_pretty_string)

    assert str(driven_control) == expected_string


def test_resample_exact():
    driven_control = DrivenControl(
        rabi_rates=np.array([0, 2]),
        azimuthal_angles=np.array([1.5, 0.5]),
        detunings=np.array([1.3, 2.3]),
        durations=np.array([1, 1]),
        name="control",
    )

    new_driven_control = driven_control.resample(0.5)

    assert len(new_driven_control.durations) == 4

    assert np.allclose(new_driven_control.durations, 0.5)
    assert np.allclose(new_driven_control.rabi_rates, [0, 0, 2, 2])
    assert np.allclose(new_driven_control.azimuthal_angles, [1.5, 1.5, 0.5, 0.5])
    assert np.allclose(new_driven_control.detunings, [1.3, 1.3, 2.3, 2.3])


def test_resample_inexact():
    driven_control = DrivenControl(
        rabi_rates=np.array([0, 2]),
        azimuthal_angles=np.array([1.5, 0.5]),
        detunings=np.array([1.3, 2.3]),
        durations=np.array([1, 1]),
        name="control",
    )

    new_driven_control = driven_control.resample(0.3)

    assert len(new_driven_control.durations) == 7

    assert np.allclose(new_driven_control.durations, 0.3)
    assert np.allclose(new_driven_control.rabi_rates, [0, 0, 0, 0, 2, 2, 2])
    assert np.allclose(
        new_driven_control.azimuthal_angles, [1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5]
    )
    assert np.allclose(
        new_driven_control.detunings, [1.3, 1.3, 1.3, 1.3, 2.3, 2.3, 2.3]
    )
