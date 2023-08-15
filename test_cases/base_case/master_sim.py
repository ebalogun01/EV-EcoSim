"""
Runs the base_case simulation without any EV charging station/DER within the power network.
"""

import gridlabd
import sys
sys.path.append('../../../EV50_cosimulation/charging_sim')
# from event_handlers import EV_Charging_sim

print('Starting GridLAB-D')
gridlabd.command("IEEE123_populated.glm")
gridlabd.start("wait")
print("started...")

# EV_Charging_sim.load_result_summary()