# Q-CTRL Open Controls
[![Actions Status](https://github.com/qctrl/python-open-controls/workflows/Push%20workflow/badge.svg)](https://github.com/qctrl/python-open-controls/actions?query=workflow%3A"Push+workflow")
[![Actions Status](https://github.com/qctrl/python-open-controls/workflows/Release%20workflow/badge.svg)](https://github.com/qctrl/python-open-controls/actions?query=workflow%3A"Release+workflow")

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

   ```bash
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

See the [Jupyter notebooks](https://github.com/qctrl/notebooks/tree/master/qctrl-open-controls).

## Contributing

For general guidelines, see [Contributing](https://github.com/qctrl/.github/blob/master/CONTRIBUTING.md).

### Building documentation

Documentation generation relies on [Sphinx](http://www.sphinx-doc.org). Automated builds are done by [Read The Docs](https://readthedocs.com).

To build locally:

1. Ensure you have used one of the install options above.
1. Execute the make file from the docs directory:

    If using Poetry:

    ```bash
    cd docs
    poetry run make html
    ```

    If using setuptools:

    ```bash
    cd docs
    # Activate your virtual environment if required
    make html
    ```

The generated HTML will appear in the `docs/_build/html` directory.

## Credits

See
[Contributors](https://github.com/qctrl/python-open-controls/graphs/contributors).

## License

See [LICENSE](LICENSE).
