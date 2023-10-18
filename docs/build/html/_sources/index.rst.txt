.. EV-Ecosim documentation master file, created by
   sphinx-quickstart on Wed Aug  9 14:16:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
Welcome to EV-Ecosim 1.0.0 documentation!
=====================================
EV-Ecosim is a python-based multi-timescale grid-aware python-based co-simulation platform for modeling and simulation
of electric vehicle (EV) charging stations integrated with Distributed Energy Resources (DERs), such as solar and
battery systems. This platform interfaces with GridLAB-D, a 3-phase unbalanced power flow solver, to capture the impacts
of EV (Fast) Charging Stations on power distribution networks.

Links
--------
- `Preprint <https://doi.org/10.36227/techrxiv.23596725.v2>`_
- `Repository <https://github.com/ebalogun01/EV50_cosimulation/>`_
-  Web-tool (coming soon!)

.. image:: ../../doc_images/sim_frame.png
    :width: 100%
    :align: center
    :alt: EV-Ecosim Simulation Framework

We have made a web-tool version available for public use <include link>. More sophisticated users can download the
source code and run the platform locally.

.. toctree::
   :hidden:
   :maxdepth: 1

   Introduction<Introduction/welcome>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API:
   api

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials:

   Tutorial<tutorials>

.. toctree::
   :hidden:
   :maxdepth: 2

   Readme<readme/README>

.. toctree::
    :hidden:
    :maxdepth: 2
    :caption: Modules:

    modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
