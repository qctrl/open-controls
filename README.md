# Q-CTRL Open Controls

Q-CTRL Open Controls is an open-source Python package that makes it easy to
create and deploy established error-robust quantum control protocols from the
open literature. The aim of the package is to be the most comprehensive library
of published and tested quantum control techniques developed by the community,
with easy to use export functions allowing users to deploy these controls on:

- Custom quantum hardware
- Publicly available cloud quantum computers
- The [Q-CTRL product suite](https://q-ctrl.com/products/)

Anyone interested in quantum control is welcome to contribute to this project.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

## Installation

Q-CTRL Open Controls can be install through `pip` or from source. We recommend
the `pip` distribution to get the most recent stable release. If you want the
latest features then install from source.

### Requirements

To use Q-CTRL Open Controls you will need an installation of Python. We
recommend using the [Anaconda](https://www.anaconda.com/) distribution of
Python. Anaconda includes standard numerical and scientific Python packages
which are optimally compiled for your machine. Follow the [Anaconda
Installation](https://docs.anaconda.com/anaconda/install/) instructions and
consult the [Anaconda User
guide](https://docs.anaconda.com/anaconda/user-guide/) to get started.

We use interactive jupyter notebooks for our usage examples. The Anaconda
python distribution comes with editors for these files, or you can [install the
jupyter notebook editor](https://jupyter.org/install) on its own.

### Using PyPi

Use `pip` to install the latest version of Q-CTRL Open Controls.

```shell
pip install qctrl-open-controls
```

### From Source

The source code is hosted on
[Github](https://github.com/qctrl/python-open-controls). The repository can be
cloned using

```shell
git clone git@github.com:qctrl/python-open-controls.git
```

Once the clone is complete, you have two options:

1. Using setup.py

   ```shell
   cd python-open-controls
   python setup.py develop
   ```

   **Note:** We recommend installing using `develop` to point your installation
   at the source code in the directory where you cloned the repository.

1. Using Poetry

   ```shell
   cd python-open-controls
   ./setup-poetry.sh
   ```

   **Note:** if you are on Windows, you'll need to install
   [Poetry](https://poetry.eustace.io) manually, and use:

   ```cmd
   cd python-open-controls
   poetry install
   ```

Once installed via one of the above methods, test your installation by running
`pytest`
in the `python-open-controls` directory.

```shell
pytest
```

## Usage

Usage depends on the application. We've provided a set of [example Jupyter
notebooks](examples) addressing a variety of quantum control problems. Below is
a short description of each notebook grouped by application. For further
details on usage, use the inline documentation in the source code.

### Dynamical Decoupling Sequences (DDS)

Q-CTRL Open Controls can create a large library of standard DDS which can be
exported in a variety of formats.

#### Create a DDS

[`examples/creating_a_dds.ipynb`](examples/creating_a_dds.ipynb) demonstrates
how to use Q-CTRL Open Controls to create a DDS from a large library of
published dynamical decoupling protocols. It also shows how to make Custom DDS
with timings, offsets and unitaries defined by the user. The notebook shows how
to export a DDS for deployment in the [Q-CTRL
products](https://q-ctrl.com/products/) or your quantum hardware.

#### Export a DDS to Qiskit

[`examples/export_a_dds_to_qiskit.ipynb`](examples/export_a_dds_to_qiskit.ipynb)
demonstrates how to take a DDS and convert it to a Qiskit circuit so it can be
run on IBM's quantum computers. It also demonstrates using a DDS to improve the
performance of a quantum circuit execution by extending the coherence time of a
qubit.

#### Export a DDS to Cirq

[`examples/export_a_dds_to_cirq.ipynb`](examples/export_a_dds_to_cirq.ipynb)
demonstrates how to take a DDS and convert it to a Cirq circuit or schdule. It
also shows how to run a circuit or schedule in a Cirq simulator.

## Contributing

See
[Contributing](https://github.com/qctrl/.github/blob/master/CONTRIBUTING.md).

## Credits

See
[Contributors](https://github.com/qctrl/python-open-controls/graphs/contributors).

## License

See [LICENSE](LICENSE).
