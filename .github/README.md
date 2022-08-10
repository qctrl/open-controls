# Q-CTRL Python Open Controls

Q-CTRL Open Controls is an open-source Python package that makes it easy to
create and deploy established error-robust quantum control protocols from the
open literature. The aim of the package is to be the most comprehensive library
of published and tested quantum control techniques developed by the community,
with easy to use export functions allowing users to deploy these controls on:

- Custom quantum hardware
- Publicly available cloud quantum computers
- The [Q-CTRL product suite](https://q-ctrl.com/products/)

Anyone interested in quantum control is welcome to contribute to this project.

## Installation

Q-CTRL Open Controls can be installed through `pip` or from source. We recommend
the `pip` distribution to get the most recent stable release. If you want the
latest features, then install from source.

### Requirements

To use Q-CTRL Open Controls you will need an installation of Python (>=3.7, <3.11).
We recommend using the [Anaconda](https://www.anaconda.com/) distribution of
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

1. Using Poetry

   Follow the instructions from the
   [Poetry documentation](https://python-poetry.org/docs/#installation) to
   install Poetry.

   After you have installed Poetry, use:

   ```bash
   cd python-open-controls
   poetry install
   ```

1. Using pip

   ```shell
   cd python-open-controls
   poetry export --dev -f requirements.txt --output requirements.txt --without-hashes
   pip install -r requirements.txt
   pip install -e .
   rm requirements.txt
   ```

Once installed via one of the above methods, test your installation by running
`pytest`
in the `python-open-controls` directory.

```shell
pytest
```

## Usage

See the [Jupyter notebooks examples](../examples) and the
[Q-CTRL Open Controls reference documentation](https://docs.q-ctrl.com/open-controls/references/qctrl-open-controls/).

## Contributing

For general guidelines, see [Contributing](https://github.com/qctrl/.github/blob/master/CONTRIBUTING.md).

### Building documentation

Documentation generation relies on [Sphinx](http://www.sphinx-doc.org).
The reference documentation for the latest released version of
Q-CTRL Open Controls is hosted online in the
[Q-CTRL documentation website](https://docs.q-ctrl.com/open-controls/references/qctrl-open-controls/).

To build it locally:

1. Ensure you have used one of the install options above.
1. Execute the make file from the docs directory:

    If using Poetry:

    ```bash
    cd docs
    poetry run make html
    ```

    If using pip:

    ```bash
    cd docs
    # Activate your virtual environment if required
    make html
    ```

The generated HTML will appear in the `docs/_build/html` directory.

### Formatting, linting, and static analysis

Code is formatted, linted and checked using the following tools:
- [Black](https://github.com/psf/black)
- [Pylint](https://pypi.org/project/pylint/)
- [isort](https://github.com/timothycrosley/isort)
- [mypy](http://mypy-lang.org/)

These checks are run on all code merged to master, and may also be run locally from the python-open-controls
directory:

```shell
pip install black pylint_runner isort mypy
mypy
isort --check
black --check .
pylint_runner
```

Black and isort, in addition to checking code, can also automatically apply fixes. To fix all code
in the python-open-controls tree, run:

```shell
isort
black .
```

You can also run these checks only in the files that you changed by using the
`pre-commit` tool. To use it, run:

```shell
pip install pre-commit
pre-commit install
```

With this, the checks will run every time that you commit code with
`git commit`. If you prefer to run the checks every time that you push changes
instead of when you commit changes, use `pre-commit install -t pre-push`.

If you no longer wish to use `pre-commit`, you can uninstall it by running
`pre-commit uninstall` in the `python-open-controls` directory (or by running
`pre-commit uninstall -t pre-push`, if you used the pre-push hooks).

## Credits

See
[Contributors](https://github.com/qctrl/python-open-controls/graphs/contributors).

## License

See [LICENSE](../LICENSE).
