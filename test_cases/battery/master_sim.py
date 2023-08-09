"""
This file kicks off the online co-simulation. Initializes GridLAB-D with the relevat GLM and runs the program
via event_handlers.py.
"""
import gridlabd
import gblvar


def run(scenario: dict):
    """Runs a scenario.
    Inputs: scenario - dict describing the scenario to be run.
    Returns: True - this is by default and arbitrary.
    """
    gblvar.scenario = scenario
    gridlabd.command("IEEE123_populated.glm")
    gridlabd.start("wait")
    return True

