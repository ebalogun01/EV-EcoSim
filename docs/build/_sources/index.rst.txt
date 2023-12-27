.. EV-EcoSim documentation master file, created by
   sphinx-quickstart on Wed Aug  9 14:16:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====================================
Welcome to *EV-EcoSim* 1.0.0 documentation!
=====================================

.. image:: ../../doc_images/sim_control_diagram.png
    :width: 100%
    :align: center
    :alt: EV-EcoSim Simulation Framework
--------------------------------

--------------------------------
What is *EV-EcoSim*?
--------------------------------
*EV-EcoSim* is a python-based multi-timescale grid-aware co-simulation platform for modeling, control, optimization,
and simulation of electric vehicle (EV) charging stations. The charging infrastructure developed within this platform is
integrated with Distributed Energy Resources (DERs), such as solar and battery systems. This platform interfaces with
GridLAB-D, a 3-phase unbalanced power flow solver, to capture the impacts of EV (Fast) Charging Stations on power
distribution networks.

Links
--------
- `Preprint <https://doi.org/10.36227/techrxiv.23596725.v2>`_
- `Repository <https://github.com/ebalogun01/EV50_cosimulation/>`_
-  Web-tool (coming soon!)

.. toctree::
   :hidden:
   :maxdepth: 1

   Introduction<Introduction/welcome>

.. toctree::
   :hidden:
   :maxdepth: 1

   Readme<readme/README>

.. toctree::
   :hidden:
   :maxdepth: 1

   Tutorial<tutorials/tutorial>

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Modules

    modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
