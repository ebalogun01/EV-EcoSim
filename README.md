# EV50 co-simulation platform

Platform for co-simulating power flow and EV charging stations for Stanford EV50 project. 


## Requirements

GiSMo SLAC GridLAB-D installation (master branch): https://github.com/slacgismo/gridlabd. Recommended use with AWS EC2 SLAC GiSMo HiPAS GridLAB-D AMI (beauharnois-X).

## Folder descriptions

### feeders

Library of IEEE test feeders and PNNL taxonomy feeders for distribution systems in the GridLAB-D .glm format.
IEEE feeders have spot loads specified at primary distribution level. PNNL taxonomy feeders have spot loads specified at primary or secondary distribution level.


### feeder_population

Scripts for populating base feeder models with time-varying loads and resources. feeder_population.py generates the necessary files for a co-simulation run based on the parameters specified in config.txt. Requires residential load data not included in repo (file size too large).


### test_cases

Co-simulation cases. 

Base_case - Reads voltage from GridLAB-D and writes power injections at each timestep.


### analysis

Scripts for plotting and analysis of co-simulation results.