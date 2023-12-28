"""
Runs the base_case simulation without any EV charging station/DER within the power network.
"""

import gridlabd
import sys
sys.path.append('../../charging_sim')

print('Starting GridLAB-D')
gridlabd.command("IEEE123_populated.glm")
gridlabd.start("wait")
print("started...")
