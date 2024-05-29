# Copyright 2024 Q-CTRL
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
Configuration file for the Sphinx documentation builder.
"""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import inspect
import os
import sys
from typing import List

import tomli

import qctrlopencontrols

# pylint:disable=invalid-name
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
with open("../pyproject.toml", "rb") as f:
    parsed = tomli.load(f)
package_info = parsed["tool"]["poetry"]
project = package_info["description"]
author = ", ".join(package_info["authors"])
release = package_info["version"]
copyright = f"{datetime.datetime.now().year} Q-CTRL. All rights reserved."  # pylint: disable=redefined-builtin

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_markdown_builder",
]

master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = []


# -- Options for HTML output -------------------------------------------------

# Option to automatically generate summaries.
autosummary_generate = True

# Hide type hints in signatures.
autodoc_typehints = "none"

public_apis = qctrlopencontrols.__all__

# the key of autosummary_context can be used
# as variable in the template
# here `qctrlopencontrols` is used in the template for
# providing a list of all public APIs
autosummary_context = {"qctrlopencontrols": public_apis}

# Builds filename/url mappings for the objects

# autosummary_filename_map allows us to customize the name of individual doc for each API
# by mapping the name from the key to the value, to provide a better URL reflecting how
# the APIs are exposed

# update file name class and function
autosummary_filename_map = {
    qctrlopencontrols.__name__ + "." + api: api for api in public_apis
}

# update file name for class methods and attributes
for _class in [
    value for name, value in inspect.getmembers(qctrlopencontrols, inspect.isclass)
]:
    autosummary_filename_map.update(
        {
            qctrlopencontrols.__name__
            + "."
            + _class.__name__
            + "."
            + attribute: _class.__name__
            + "."
            + attribute
            for attribute in dir(_class)
            if not attribute.startswith("_")
        }
    )
