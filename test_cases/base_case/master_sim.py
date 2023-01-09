# import os
# import numpy as np
import gridlabd
# from event_handlers import EV_Charging_sim

print('Ok')
gridlabd.command("IEEE123_populated.glm")
gridlabd.start("wait")
print("started...")

# EV_Charging_sim.load_result_summary()