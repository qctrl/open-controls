Installation
============

Q-CTRL Open Controls can be install through ``pip`` or from source. We recommend
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

Use ``pip`` to install the latest version of the Q-CTRL ppen controls Python package.

.. code-block:: shell

   pip install qctrl-open-controls

From Source
-----------

The source code is hosted on
`Github <https://github.com/qctrl/python-open-controls>`_. The repository can be
cloned using

.. code-block:: shell

   git clone git@github.com:qctrl/python-open-controls.git

Once the clone is complete, you have two options:


#. 
   Using setup.py

   .. code-block:: shell

      cd python-open-controls
      python setup.py develop

   **Note:** We recommend installing using ``develop`` to point your installation
   at the source code in the directory where you cloned the repository.

#. 
   Using Poetry

   .. code-block:: shell

      cd python-open-controls
      ./setup-poetry.sh

   **Note:** if you are on Windows, you'll need to install
   `Poetry <https://poetry.eustace.io>`_ manually, and use:

   .. code-block:: shell

      cd python-open-controls
      poetry install

Once installed via one of the above methods, test your installation by running
``pytest``
in the ``python-open-controls`` directory.

.. code-block:: shell

   pytest
