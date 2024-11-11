# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Patra Model Card Toolkit'
copyright = '2024, Data To Insights Center, Indiana University'
author = 'Data To Insights Center, Indiana University'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',   # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for Google-style docstrings
]
import os
import sys
sys.path.insert(0, os.path.abspath('/Users/neeleshkarthikeyan/d2i/patra-toolkit/patra_model_card'))

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
