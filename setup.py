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
Setup script for Q-CTRL Open Controls
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from setuptools import setup, find_packages


def read_version():
    """Reads the version
    """
    version={}
    with open('qctrlopencontrols/version.py') as version_file:
        exec(version_file.read(), version)
    return version


def read_license():
    """
    Reads the LICENSE file.
    """
    with open('LICENSE') as license_file:
        return license_file.read()


def main():
    version = read_version()
    setup(
        name='qctrl-open-controls',
        version=version['__version__'],
        packages=find_packages(),
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=['numpy', 'scipy', 'pytest', 'nbval', 'qiskit'],
        author='Q-CTRL',
        author_email='support@q-ctrl.com',
        description='Q-CTRL Open Controls',
        license='Apache-2.0',
        keywords='quantum, computing, open source, engineering',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Visualization',
            'Topic :: Software Development :: Embedded Systems',
            'Topic :: System :: Distributed Computing'
        ],
    )


if __name__ == '__main__':
    main()
