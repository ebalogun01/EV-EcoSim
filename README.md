# EV-Ecosim platform

A grid-aware co-simulation platform for the design and optimization of electric vehicle charging stations. 
Paper: https://doi.org/10.36227/techrxiv.23596725.v2

## Authors
Emmanuel Balogun: ebalogun@stanford.edu, Lily Buechler: ebuech@stanford.edu


## Requirements

GiSMo SLAC GridLAB-D installation (master branch): https://github.com/slacgismo/gridlabd. This GridLAB-D version is required for the python co-simulation functionality. Recommended use with AWS EC2 SLAC GiSMo HiPAS GridLAB-D AMI (beauharnois-X).

## Folder descriptions

### ambient_data

Hosts ambient temperature data for capturing the effects of environmental conditions on subsystems, such as battery, 
transformers, charging stations.


### base_load_data

Includes existing base case building/home load (usually uncontrollable) within the distribution grid. This work uses 
proprietary Pecan Street Data.


### charging_sim

Hosts the implementation of the physical modules, including:
##### `battery.py` - Battery cell module. 
##### `batterypack.py` - Battery pack module.
##### `batteryAgingSim.py` - Battery aging module.
##### `controller.py` - Controller module.
##### `chargingStation.py` - Charging station module.
##### `electricityPrices.py` - Electricity prices module. 
##### `optimization.py` - Optimization module.
##### `orchestrator.py` - Simulation orchestrator module.
##### `solar.py` - Solar PV module.
##### `utils.py` - Hosts utility functions used by some modules.
##### `simulate.py` - Offline DER control optimization for cost minimization (this is run for offline mode (no state feedback)).

It also hosts the `configs` folder which includes the configuration files for all the relevant modules.


### DLMODELS

This includes legacy load forecasts models developed (not needed).


### elec_rates

Includes .csv files for electricity Time-of-use (TOU) rates.


### feeders

Library of IEEE test feeders and PNNL taxonomy feeders for distribution systems in the GridLAB-D `.glm` format.
IEEE feeders have spot loads specified at primary distribution level. PNNL taxonomy feeders have spot loads specified at
primary or secondary distribution level.


### feeder_population

Scripts for populating base feeder models with time-varying loads and resources using the load data in base_load_data. 
`feeder_population.py` generates the necessary files for a co-simulation run based on the parameters specified in 
`feeder_population/config.txt`. Requires residential load data not included in repo (limited access).


### test_cases

Co-simulation cases. 

##### `base_case`- Reads voltage from GridLAB-D and writes power injections at each timestep (no EV charging or DER).  
##### `rlsf` - base_case plus implements a recursive least squares filter to estimate network model online (not used).
##### `battery` - base_case plus transformer thermal model plus DER integration (included battery and solar).
##### `transformer` - base_case plus simulation of transformer thermal model for each transformer in GridLAB-D model (not used).



### analysis

Scripts for plotting and analysis of co-simulation results. Includes post optimization and simulation cost 
calculation modules and voltage impacts on the distribution grid.

`plot_results.py` - plot voltage profiles from simulation, save plots, calculate % violations per ANSI standards.
`plot_rlsf_error.py` - plot online prediction error from recursive least squares filter model of power flow.  


## How to run
Create a new environment using `conda env create --name <your env name> -f environment.yml`OR 
install packages listed in the environment manually (RECOMMENDED)
Ensure gridlabd is installed by following recommended installation method.
Start simulation by running master_sim.py.

