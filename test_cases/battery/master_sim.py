import os
import time

import gridlabd
import gblvar

# MODIFY CONFIGURATION FILES
# global scenario_map


def run(scenario):
    gblvar.scenario = scenario
    # print("scenario", scenario)
    # if scenario['index'] == 0:
    # gridlabd.cancel()
    gridlabd.command("IEEE123_populated.glm")
    # print("YESSS", os.getcwd()+'/sim_'+str(scenario['index']))
    gridlabd.start("wait")
    # gridlabd.set_value("voltdump", "filename", os.getcwd() + '/sim_' + str(scenario['index']))

    return True
        # gridlabd.cancel()
    # else:
    #     gridlabd.start()

