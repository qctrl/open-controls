#!/bin/bash

echo "--- Checking appropriate prerequisites are installed."

# Check correct version of Python
python -V | grep 3.7
if [ $? -ne 0 ]
then
    echo "Please ensure you have Python 3.7 activated. Pyenv is recommended."
    exit 1
fi

# Check Poetry installed
poetry --version | grep 0.12
if [ $? -ne 0 ]
then
    echo "--- Poetry not detected, installing..."
    curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
fi

# Do dry run to create virtual environment
echo "--- Using Poetry to create virtual environment"
poetry install --dry-run -q

# Activate created virtual environment
echo "--- Activating virtual environment"
VIRTUAL_ENV=$(poetry run python  -c "import os; print(os.environ['VIRTUAL_ENV'])")
echo "Virtual environment path is $VIRTUAL_ENV"
source $VIRTUAL_ENV/bin/activate
if [ $? -ne 0 ]
then
    echo "Could not activate the virtual environment!"
    exit 2
fi

echo "--- Updating pip and setuptools"
pip install -q --upgrade pip setuptools
deactivate

# Do the final install
echo "--- Installing dependencies"
poetry update