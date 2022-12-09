import gridlabd
import gblvar

# MODIFY CONFIGURATION FILES


def run(scenario):
    gblvar.scenario = scenario
    gridlabd.command("IEEE123_populated.glm")
    gridlabd.start("wait")

