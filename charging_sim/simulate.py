"""
This module runs the optimization offline without the power system or battery state feedback for each time-step.
This is done to save time. Once this is done, one can study the effects on the power system. Power system states are
propagated post optimization to fully characterize what would have occurred if in-situ optimization was done.
"""

import os
from orchestrator import ChargingSim
import multiprocessing as mp
import numpy as np
import json
import ast
from utils import month_days

# GET THE PATH PREFIX FOR SAVING THE INPUTS
path_prefix = os.getcwd()
path_prefix = path_prefix[: path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
# month = 6  # month index starting from 1. e.g. 1: January, 2: February, 3: March etc.

day_minutes = 1440
opt_time_res = 15  # minutes
num_days = 30  # determines optimization horizon
num_steps = num_days * day_minutes // opt_time_res  # number of steps to initialize variables for opt

# PRELOAD
station_config = open(path_prefix + '/test_cases/battery/feeder_population/config.txt', 'r')
param_dict = station_config.read()
station_config.close()
param_dict = ast.literal_eval(param_dict)
L2_station_cap = float(param_dict['l2_charging_stall_base_rating'].split('_')[0]) * param_dict['num_l2_stalls_per_node']
dcfc_station_cap = float(param_dict['dcfc_charging_stall_base_rating'].split('_')[0]) * param_dict[
    'num_dcfc_stalls_per_node']
month = int(str(param_dict['starttime']).split('-')[
                1])  # month index starting from 1. e.g. 1: January, 2: February, 3: March etc.
month_str = list(month_days.keys())[month - 1]

# Load DCFC locations txt file.
print('...loading charging bus nodes')
dcfc_nodes = np.loadtxt('../test_cases/battery/dcfc_bus.txt', dtype=str).tolist()  # This is for DC FAST charging.
if type(dcfc_nodes) is not list:
    dcfc_nodes = [dcfc_nodes]
dcfc_dicts_list = []
for node in dcfc_nodes:
    dcfc_dicts_list += {"DCFC": dcfc_station_cap, "L2": 0, "node": node},

L2_charging_nodes = np.loadtxt('../test_cases/battery/L2charging_bus.txt',
                               dtype=str).tolist()  # this is for L2 charging
if type(L2_charging_nodes) is not list:
    L2_charging_nodes = [L2_charging_nodes]
l2_dicts_list = []
for node in L2_charging_nodes:
    l2_dicts_list += {"DCFC": 0, "L2": L2_station_cap, "node": node},
num_charging_nodes = len(dcfc_nodes) + len(
    L2_charging_nodes)  # needs to come in as input initially & should be initialized prior from the feeder population

#   RUN TYPE
sequential_run = True
parallel_run = False
single_run = False

# BATTERY SCENARIOS
num_vars = 6
min_power = 0
max_power = 0
power_ratings = []  # this should be redundant for max_c_rate
energy_ratings = [5e4, 10e4, 20e4, 40e4, 80e4]
# ENERGY_RATINGS = [0]
# MAX_C_RATES = [0.00001]
max_c_rates = [0.1, 0.2, 0.5, 1, 2]
min_SOCs = [0.1, 0.2, 0.3]
max_SOCs = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


def make_scenarios():
    """
    This is used to make the list of scenarios (dicts) that are used to run the simulations.
    No inputs. However, it uses preloaded global functions from a `config.txt` file.

    :return: None.
    """

    scenarios_list = []
    idx = 0
    for Er in energy_ratings:
        for c_rate in max_c_rates:
            scenario = {'pack_energy_cap': Er, 'max_c_rate': c_rate, 'index': idx, 'opt_solver': 'GUROBI',
                        'oneshot': True, 'start_month': month}
            scenarios_list.append(scenario)
            idx += 1
    return scenarios_list


def run(scenario):
    """
    Runs a scenario and updates the scenario JSON to reflect main properties of that scenario.

    :param scenario: The scenario dictionary that would be run.
    :return: None. Runs the `scenario`.
    """
    EV_charging_sim = ChargingSim(num_charging_nodes, path_prefix=path_prefix, num_steps=num_steps, month=month)
    save_folder_prefix = f'oneshot_{month_str}{str(scenario["index"])}/'
    os.mkdir(save_folder_prefix)
    EV_charging_sim.setup(dcfc_dicts_list + l2_dicts_list, scenario=scenario)
    EV_charging_sim.multistep()
    EV_charging_sim.load_results_summary(save_folder_prefix)
    with open(f'{save_folder_prefix}scenario.json', "w") as outfile:
        json.dump(scenario, outfile)


def run_scenarios_parallel():
    """
    Runs the scenarios in parallel using the `multiprocessing` library. User should have enough cores and RAM,
    otherwise, this may lead to entire process freezing.

    :return: None
    """
    scenarios = make_scenarios()
    start_idx = 0
    end_idx = 10
    num_cores = mp.cpu_count()
    if num_cores > 1:
        use_cores_count = min(num_cores - 2, end_idx - start_idx)  # leave one out
        print(f"Running {use_cores_count} parallel scenarios...")
        with mp.get_context("spawn").Pool(use_cores_count) as pool:
            pool.map(run, [scenarios[i] for i in range(start_idx, min(use_cores_count + start_idx, end_idx))])


def run_scenarios_sequential():
    """
    Creates scenarios based on the energy and c-rate lists/vectors and runs each of the scenarios,
    which is a combination of all the capacities and c-rates.

    :return: None.
    """
    start_idx = 0
    end_idx = len(energy_ratings) * len(max_c_rates)
    idx_list = list(range(start_idx, end_idx))
    scenarios_list = make_scenarios()
    scenarios = [scenarios_list[idx] for idx in idx_list]
    # d = 1
    for scenario in scenarios:
        scenario["L2_nodes"] = L2_charging_nodes
        scenario["dcfc_nodes"] = dcfc_nodes
        if dcfc_dicts_list:
            scenario["dcfc_caps"] = [station["DCFC"] for station in dcfc_dicts_list]
        if l2_dicts_list:
            scenario["l2_caps"] = [station["L2"] for station in l2_dicts_list]
        run(scenario)
        # d += 1


def run_scenario_single():
    """
    Runs a single scenario dict.

    :return: None.
    """
    # Keep changing this for each run
    Er_idx = 0
    c_rate_idx = 2
    idx = 2
    scenario = {'pack_energy_cap': energy_ratings[Er_idx],
                'max_c_rate': max_c_rates[c_rate_idx],
                'index': idx, 'opt_solver': 'GUROBI', 'oneshot': True}
    run(scenario)


if __name__ == '__main__':
    if sequential_run:
        run_scenarios_sequential()
    elif single_run:
        run_scenario_single()
    else:
        run_scenarios_parallel()
