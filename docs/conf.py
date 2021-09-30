# Copyright 2021 Q-CTRL. All rights reserved.
#
# Licensed under the Q-CTRL Terms of service (the "License"). Unauthorized
# copying or use of this file, via any medium, is strictly prohibited.
# Proprietary and confidential. You may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#      https://q-ctrl.com/terms
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS. See the
# License for the specific language.
# pylint:disable=invalid-name

"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Path setup --------------------------------------------------------------

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.
"""

import datetime
import inspect
import os
import sys
from typing import List

import toml

import qctrlopencontrols

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
parsed = toml.load("../pyproject.toml")
package_info = parsed["tool"]["poetry"]
project = package_info["description"]
author = ", ".join(package_info["authors"])
release = package_info["version"]
copyright = f"{datetime.datetime.now().year} Q-CTRL. All rights reserved."  # pylint: disable=redefined-builtin

# -- HTML Variables ----------------------------------------------------------
# Variables to insert into _templates/layout.html
# Taken from qctrl/docs, see head.html in the repo
html_context = {
    "var_url": "https://docs.q-ctrl.com",
    "var_title": "Open Controls Python package | Q-CTRL",
    "var_description": (
        "Module, class and method reference for the "
        "Q-CTRL Open Controls Python package"
    ),
    "var_image": (
        "https://images.ctfassets.net/l5sdcktfe9p6/6XwDBOS4cWBv2Lo"
        "SimZwKj/1fb29e7c1e2941114d5eda81797e84ac/q-ctrl.jpg"
    ),
    "var_twitter_username": "qctrlHQ",
}

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
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
]

master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_title = html_context["var_title"]

# Theme options
html_theme_options = {
    # Toc options
    "collapse_navigation": False,
    "includehidden": False,
}

# Option to automatically generate summaries.
autosummary_generate = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "https://docs.q-ctrl.com/assets/css/readthedocs.css",
]
html_js_files = [
    "https://docs.q-ctrl.com/assets/js/readthedocs.js",
]
html_logo = "_static/logo.svg"


public_apis = qctrlopencontrols.__all__

# the key of autosummary_context can be used
# as variable in the template
# here `qctrlopencontrols` is used in the template for
# providing a list of all public APIs
autosummary_context = {
    "qctrlopencontrols": public_apis,
}

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
