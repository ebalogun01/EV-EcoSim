import gridlabd
import gblvar


def run(scenario):
    gblvar.scenario = scenario
    gridlabd.command("IEEE123_populated.glm")
    gridlabd.start("wait")
    return True

