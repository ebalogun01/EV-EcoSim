# EV-Ecosim platform

A grid-aware co-simulation platform for the design and optimization of electric vehicle charging stations. 
Paper: https://doi.org/10.36227/techrxiv.23596725.v2

![sim_frame.png](doc_images%2Fsim_frame.png)

[//]: # (<img src="doc_images/sim_frame.png" alt="EV-Ecosim Framework Description" width="3000" height="400" title="EV-Ecosim Framework Description">)

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

#TODO: add base load data prototype


### batt_sys_identification
Battery system identification module. Hosts the class for generating battery system identification parameters
from experimental data. This module leverages a genetic algorithm to optimize the battery model parameters. 
The battery model is a 2nd order RC Equivalent circuit model (ECM). One can this module to generate custom NMC
battery parameters by uploading experimental data to the `batt_sys_identification/data` folder and running the module.
The module will generate a `.csv` file with the battery parameters in the `batt_sys_identification/params` folder.
The data prototype is shown below. Note that column fields are case-sensitive.

![batt_sys_data_proto.png](doc_images%2Fbatt_sys_data_proto.png)

The module will save a new `.csv` file with an additional field for the corrected open circuit voltage (OCV) values;
this field (column) will be labelled `ocv_corr` within the new battery data csv, including the existing columns as shown
in the data prototype above.

Once the battery parameters are generated, they can be used in the `battery_data` folder and `configs/battery.json` can 
be modified so the model runs using the new custom parameters.

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


### solar_data
Includes solar irradiance data for capturing the effects of environmental conditions on overall system cost. Default
data for solar irradiance is from the National Solar Radiation Database (NSRDB) for the San Francisco Bay Area.
The data prototype is from the National Renewable Energy Laboratory (NREL) and is shown below. Note that column fields
are case-sensitive.

![solar_data_proto.png](doc_images%2Fsolar_data_proto.png)

### test_cases

#### Co-simulation cases. 

 `base_case`- Reads voltage from GridLAB-D and writes power injections at each timestep (no EV charging or DER).

`rlsf` - base_case plus implements a recursive least squares filter to estimate network model online (not used)

`battery` - base_case plus transformer thermal model plus DER integration (included battery and solar).

`transformer` - base_case plus simulation of transformer thermal model for each transformer in GridLAB-D model (not used).



### analysis

Scripts for plotting and analysis of co-simulation results. Includes post optimization and simulation cost 
calculation modules and voltage impacts on the distribution grid.

`plot_results.py` - plot voltage profiles from simulation, save plots, calculate % violations per ANSI standards.

`plot_rlsf_error.py` - plot online prediction error from recursive least squares filter model of power flow.  


## How to run
Create a new environment using `conda env create --name <your env name> -f environment.yml`OR 
install packages listed in the environment manually (RECOMMENDED)
Ensure gridlabd is installed by following recommended installation method.

For battery test case:
* Navigate to `test_cases/battery/feeder_population` and run `feeder_population.py`. This uses the 
  `test_cases/battery/feeder_population/config.txt` settings to prepare the power system and populate the secondary
  distribution network with time-varying base loads, EV charging stations, with the required transformers.
* Once confirmed that `feeder_population.py` has run successfully and generates the required `IEEE123_secondary.glm` and
  `IEEE123_populated.glm` files, you are done with the initial pre-simulation run prep.
* Now navigate one level of out `/feeder_population` and run scenarios.py using `python3 scenarios.py`

For base case:
* Navigate to `EV50_cosimulation/feeder_population` and run `feeder_population.py`. This uses the 
  `./feeder_population/config.txt` settings to prepare the power system and populate the secondary distribution network \
  with time-varying base loads
* Navigate to `test_cases/base_case` and run master_sim.py using `python3 master_sim.py`

## Post-simulation analysis
 * TODO

