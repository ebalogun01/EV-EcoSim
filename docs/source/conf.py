# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.append('../../EV50_cosimulation/charging_sim')
sys.path.append('../../EV50_cosimulation/charging_sim/doc_images')
sys.path.append('../../EV50_cosimulation')

project = 'EV-Ecosim'
copyright = '2023, Emmanuel Balogun, Stanford University S3L Lab'
author = 'Emmanuel Balogun'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'recommonmark']
mathjax_path = 'http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', 'DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_sourcelink = False
