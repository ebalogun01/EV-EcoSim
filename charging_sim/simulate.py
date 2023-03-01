import os
from EVCharging import ChargingSim
import multiprocessing as mp
import numpy as np
import sys

"""This runs optimization offline without the power system or battery feedback. This is done to save time. Power 
system states are then propagated post optimization to fully characterize what would have occurred if in-situ 
optimization was done """

month = 6   # month index starting from 1. e.g. 1: January, 2: February, 3: March etc.
day_minutes = 1440
opt_time_res = 15   # minutes
num_days = 30   # determines optimization horizon
num_steps = num_days * day_minutes//opt_time_res    # number of steps to initialize variables for opt

# lood DCFC locations txt file
print('...loading charging bus nodes')
dcfc_nodes = np.loadtxt('../test_cases/battery/dcfc_bus.txt', dtype=str).tolist()     # this is for DC FAST charging
if type(dcfc_nodes) is not list:
    dcfc_nodes = [dcfc_nodes]
L2_charging_nodes = np.loadtxt('../test_cases/battery/L2charging_bus.txt', dtype=str).tolist()    # this is for L2 charging
if type(L2_charging_nodes) is not list:
    L2_charging_nodes = [L2_charging_nodes]
num_charging_nodes = len(dcfc_nodes) + len(L2_charging_nodes)  # needs to come in as input initially & should be initialized prior from the feeder population

# GET THE PATH PREFIX FOR SAVING THE INPUTS
path_prefix = os.getcwd()
path_prefix = path_prefix[: path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'

EV_charging_sim = ChargingSim(num_charging_nodes, path_prefix=path_prefix, num_steps=num_steps)

# RUN TYPE
sequential_run = False
parallel_run = False
single_run = True

# BATTERY SCENARIOS
num_vars = 6
min_power = 0
max_power = 0
power_ratings = []  # this should be redundant for max_c_rate
energy_ratings = [8e4, 10e4, 15e4, 20e4, 25e4]
max_c_rates = [0.2, 0.5, 1, 1.5, 2]
min_SOCs = [0.1, 0.2, 0.3]
max_SOCs = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


def make_scenarios():
    scenarios_list = []
    idx = 0
    for Er in energy_ratings:
        for c_rate in max_c_rates:
            scenario = {'pack_energy_cap': Er, 'max_c_rate': c_rate, 'index': idx, 'opt_solver': 'GUROBI', 'oneshot': True}
            scenarios_list.append(scenario)
            idx += 1
    return scenarios_list


def run(scenario):
    save_folder_prefix = 'test_June' + str(scenario['index']) + '/'
    os.mkdir(save_folder_prefix)
    EV_charging_sim.setup(dcfc_nodes + L2_charging_nodes, scenario=scenario)
    EV_charging_sim.multistep()
    EV_charging_sim.load_results_summary(save_folder_prefix)


def run_scenarios_parallel():
    scenarios = make_scenarios()
    start_idx = 0
    end_idx = 10
    num_cores = mp.cpu_count()
    if num_cores > 1:
        use_cores_count = min(num_cores - 2, end_idx - start_idx)  # leave one out
        print(f"Running {use_cores_count} parallel scenarios...")
        with mp.get_context("spawn").Pool(use_cores_count) as pool:
            pool.map(run, [scenarios[i] for i in range(start_idx, min(use_cores_count+start_idx, end_idx))])


def run_scenarios_sequential():
    start_idx = 0
    end_idx = 25
    idx_list = list(range(start_idx, end_idx))
    scenarios_list = make_scenarios()
    scenarios = [scenarios_list[idx] for idx in idx_list]
    for scenario in scenarios:
        process = mp.get_context('spawn').Process(target=run, args=(scenario,))
        process.start()
        process.join()


def run_scenario_single():
    """This function just runs one scenario"""
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
