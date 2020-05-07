# pylint: disable=invalid-name
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
#
import os
import sys
import toml

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
parsed = toml.load("../pyproject.toml")
package_info = parsed["tool"]["poetry"]
project = package_info["description"]
author = ", ".join(package_info["authors"])
release = package_info["version"]
copyright = "2020 Q-CTRL Pty Ltd & Q-CTRL Inc. All rights reserved."  # pylint: disable=redefined-builtin

# -- HTML Variables ----------------------------------------------------------
# Variables to insert into _templates/layout.html
# Taken from qctrl/docs, see head.html in the repo
html_context = {
    "var_url": "https://docs.q-ctrl.com",
    "var_title": "Open Controls Python package | Q-CTRL",
    "var_description": """Module, class and method reference for the
                          Q-CTRL Open Controls Python package""",
    "var_image": """https://images.ctfassets.net/l5sdcktfe9p6/6XwDBOS4cWBv2Lo
                    SimZwKj/1fb29e7c1e2941114d5eda81797e84ac/q-ctrl.jpg""",
    "var_twitter_username": "qctrlHQ",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_title = html_context["var_title"]

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
