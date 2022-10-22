# EV50 co-simulation platform

Platform for co-simulating power flow and EV charging stations for Stanford EV50 project. 

## Authors
Emmanuel Balogun: ebalogun@stanford.edu, Lily Buechler: ebuech@stanford.edu


## Requirements

GiSMo SLAC GridLAB-D installation (master branch): https://github.com/slacgismo/gridlabd. This GridLAB-D version is required for the python co-simulation functionality. Recommended use with AWS EC2 SLAC GiSMo HiPAS GridLAB-D AMI (beauharnois-X).

## Folder descriptions

### feeders

Library of IEEE test feeders and PNNL taxonomy feeders for distribution systems in the GridLAB-D .glm format.
IEEE feeders have spot loads specified at primary distribution level. PNNL taxonomy feeders have spot loads specified at primary or secondary distribution level.


### feeder_population

Scripts for populating base feeder models with time-varying loads and resources. feeder_population.py generates the necessary files for a co-simulation run based on the parameters specified in config.txt. Requires residential load data not included in repo (file size too large).


### test_cases

Co-simulation cases. 

base_case - Reads voltage from GridLAB-D and writes power injections at each timestep.  
rlsf - base_case plus implements a recursive least squares filter to estimate network model online.  
transformer - base_case plus simulation of transformer thermal model for each transformer in GridLAB-D model.
battery - base_case plus transformer model plus battery model (not finished yet)

## How to run
Create a new environment using `conda env create --name <your env name> -f environmentl.yml`
Ensure gridlabd is installed by following recommended installation method.
Start simulation by running master_sim.py.

### analysis

Scripts for plotting and analysis of co-simulation results.

plot_results.py - plot voltage profiles from simulation  
plot_rlsf_error.py - plot online prediction error from recursive least squares filter model of power flow.  
