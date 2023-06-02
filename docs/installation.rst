Installation
============

Q-CTRL Open Controls can be installed through ``pip`` or from source. We recommend
the ``pip`` distribution to get the most recent stable release. If you want the
latest features then install from source.

Requirements
------------

To use Q-CTRL Open Controls you will need an installation of Python. We
recommend using the `Anaconda <https://www.anaconda.com/>`_ distribution of
Python. Anaconda includes standard numerical and scientific Python packages
which are optimally compiled for your machine. Follow the `Anaconda
Installation <https://docs.anaconda.com/anaconda/install/>`_ instructions and
consult the `Anaconda User
guide <https://docs.anaconda.com/anaconda/user-guide/>`_ to get started.

We use interactive jupyter notebooks for our usage examples. The Anaconda
python distribution comes with editors for these files, or you can `install the
jupyter notebook editor <https://jupyter.org/install>`_ on its own.

From PyPi
---------

Use ``pip`` to install the latest version of the Q-CTRL Open Controls Python package.

.. code-block:: shell

   pip install qctrl-open-controls

From Source
-----------

The source code is hosted on
`Github <https://github.com/qctrl/open-controls>`_. The repository can be
cloned using

.. code-block:: shell

   git clone git@github.com:qctrl/open-controls.git

Once the clone is complete, you have two options:


#.
   Using Poetry

   Follow the instructions from the
   `Poetry documentation <https://python-poetry.org/docs/#installation>`_ to
   install Poetry.

   After you have installed Poetry, use:

   .. code-block:: shell

      cd open-controls
      poetry install

#.
   Using pip

   .. code-block:: shell

      cd open-controls
      poetry export --dev -f requirements.txt --output requirements.txt --without-hashes
      pip install -r requirements.txt
      pip install -e .
      rm requirements.txt

Once installed via one of the above methods, test your installation by running
``pytest`` in the ``open-controls`` directory.

.. code-block:: shell

   pytest
