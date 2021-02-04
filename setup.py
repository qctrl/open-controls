
# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


import os.path

readme = ''
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'DESCRIPTION.rst')
if os.path.exists(readme_path):
    with open(readme_path, 'rb') as stream:
        readme = stream.read().decode('utf8')


setup(
    long_description=readme,
    name='qctrl-open-controls',
    version='7.0.1',
    description='Q-CTRL Open Controls',
    python_requires='<3.9,>=3.6.4',
    project_urls={"documentation": "https://docs.q-ctrl.com/references/python/qctrl-open-controls/", "homepage": "https://q-ctrl.com", "repository": "https://github.com/qctrl/python-open-controls"},
    author='Q-CTRL',
    author_email='support@q-ctrl.com',
    license='Apache-2.0',
    keywords='q-ctrl qctrl quantum control',
    classifiers=['Development Status :: 5 - Production/Stable', 'Environment :: Console', 'Intended Audience :: Developers', 'Intended Audience :: Education', 'Intended Audience :: Science/Research', 'Natural Language :: English', 'Operating System :: OS Independent', 'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.7', 'Programming Language :: Python :: 3.8', 'Topic :: Internet :: WWW/HTTP', 'Topic :: Scientific/Engineering :: Physics', 'Topic :: Scientific/Engineering :: Visualization', 'Topic :: Software Development :: Embedded Systems', 'Topic :: System :: Distributed Computing'],
    packages=['qctrlopencontrols', 'qctrlopencontrols.driven_controls', 'qctrlopencontrols.dynamic_decoupling_sequences'],
    package_dir={"": "."},
    package_data={},
    install_requires=['numpy==1.*,>=1.16.0', 'scipy==1.*,>=1.3.0', 'toml==0.*,>=0.10.0'],
    extras_require={"dev": ["black==20.*,>=20.8.0.b1", "isort==5.*,>=5.7.0", "mypy==0.*,>=0.800.0", "nbval==0.*,>=0.9.5", "pylint==2.*,>=2.6.0", "pylint-runner==0.*,>=0.5.4", "pytest==5.*,>=5.0.0", "qctrl-visualizer==2.*,>=2.3.0", "sphinx==3.*,>=3.2.1", "sphinx-rtd-theme==0.*,>=0.4.3"]},
)
