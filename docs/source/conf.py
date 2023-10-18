# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import shutil

sys.path.insert(0, os.path.abspath("../.."))

sys.path.append('../../EV50_cosimulation/charging_sim')

# sys.path.append('../../EV50_cosimulation/charging_sim')
sys.path.append('../../EV50_cosimulation/docs/source/doc_images')

project = 'EV-Ecosim'
copyright = '2023, Emmanuel Balogun'
author = 'Emmanuel Balogun'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'recommonmark']

mathjax_path = 'http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', 'DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
master_doc = "contents"
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_sourcelink = False

# def copy_examples(app, docname):
#     if app.builder.name == 'html':
#         output_dir = os.path.join(app.outdir, 'readme', 'doc_images')
#         source_dir = os.path.join(app.srcdir, 'readme', 'doc_images')
#         print(output_dir, source_dir,"OK")
#         shutil.copytree(source_dir, output_dir)
#
#
# def setup(app):
#     app.connect('build-finished', copy_examples)