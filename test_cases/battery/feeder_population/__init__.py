"""
**Introduction**\n
This is the feeder population module within the battery test case. This file performs the pre-simulation step for
running EV-Ecosim.\n\n
It takes in a base Gridlab-D Model (GLM) file (for example, `IEEE123.glm`), and modifies that file by including
secondary distribution, home loads, and EV Charging station and transformers.


Once this script is done running, it reads and writes new GLM as <initial_glm_name>_populated.glm and
<initial_glm_name>_secondary.glm, and saves them within the test case folder. These saved files are used to run the
simulation. These files are saved in the 'test_case_dir' field specified in config.txt.


**Input file description** \n
Config `config.txt`: configuration file describing the pre-simulation parameters.
This can be modified directly or with the help of our Graphic User Interface (GUI). The return outputs of this module
are files that are read in to run the EV-Ecosim environment.


**Output file description**\n
`real_power.csv` - Real power; this is residential real load timeseries file per node/bus
`reactive_power.csv` - Reactive power; this is residential reactive load timeseries file per node/bus
`dcfc_bus.txt` - DC fast charging bus locations; this is used in co-simulation
`L2charging_bus.txt` - L2 charging bus locations; this is used in co-simulation.

"""