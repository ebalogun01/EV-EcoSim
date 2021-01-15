# EV50 co-simulation platform

Platform for co-simulating power flow and EV charging stations for Stanford EV50 project. 


## Requirements

GiSMo SLAC GridLAB-D installation (master branch): https://github.com/slacgismo/gridlabd

## Folder descriptions

### feeders

Library of IEEE test feeders and PNNL taxonomy feeders for distribution systems in the GridLAB-D .glm format.
IEEE feeders have spot loads specified at primary distribution level. PNNL taxonomy feeders have spot loads specified at primary or secondary distribution level.


### feeder_population

Scripts for populating base feeder models with time-varying loads and resources. feeder_population.py generates the necessary files for a co-simulation run based on the parameters specified in config.txt.


### test_cases

Co-simulation cases.