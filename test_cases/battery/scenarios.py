"""This file is mainly used to run scenarios in parallel (if enough CPUs) or sequentially,
 without having to modify and rerun the simulation each time.
 Currently, user defines the battery Capacities (Wh) they want to compare and the maximum allowable battery C-rates"""

import multiprocessing as mp
import sys
import gblvar
import ast

# import time

if not gblvar.charging_sim_path_append:
    sys.path.append('../../../EV50_cosimulation/charging_sim')  # change this
    gblvar.charging_sim_path_append = True
    # print('append 1')
from utils import month_days

# GET STATION CONFIGURATIONS
station_config = open('feeder_population/config.txt', 'r')
param_dict = station_config.read()
station_config.close()
param_dict = ast.literal_eval(param_dict)
L2_station_cap = float(param_dict['l2_charging_stall_base_rating'].split('_')[0]) * param_dict['num_l2_stalls_per_node']
dcfc_station_cap = float(param_dict['dcfc_charging_stall_base_rating'].split('_')[0]) * param_dict['num_dcfc_stalls_per_node']
start_month = int(str(param_dict['starttime']).split('-')[1])
month_str = list(month_days.keys())[start_month-1]

# RUN TYPE
sequential_run = True
parallel_run = False
single_run = False

# BATTERY SCENARIOS
num_vars = 6
min_power = 0
max_power = 0
power_ratings = []  # this should be redundant for max_c_rate
# month = 6
energy_ratings = [5e4, 10e4, 20e4, 40e4, 80e4]
max_c_rates = [0.1, 0.2, 0.5, 1, 2]
min_SOCs = [0.1, 0.2, 0.3]
max_SOCs = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


def make_scenarios():
    """This is used to make the list of scenarios (dicts) that are used to run the simulations"""
    scenarios_list = []
    idx = 0
    for Er in energy_ratings:
        for c_rate in max_c_rates:
            scenario = {'pack_energy_cap': Er, 'max_c_rate': c_rate, 'index': idx, 'start_month': start_month,
                        'L2_cap': L2_station_cap, 'dcfc_cap': dcfc_station_cap, 'month_str': month_str}
            scenarios_list.append(scenario)
            idx += 1
    return scenarios_list


def run(scenario):
    import master_sim
    master_sim.run(scenario)


def run_scenarios_parallel():
    """This runs c-rate-energy scenarios in parallel, using the multi-core processor of the PC.
    User should have enough cores and RAM, as if not enough, can lead to entire process freezing"""
    scenarios = make_scenarios()
    start_idx = 0
    end_idx = 3
    num_cores = mp.cpu_count()
    if num_cores > 1:
        use_cores_count = min(num_cores - 2, end_idx - start_idx)  # leave one out
        print(f"Running {use_cores_count} parallel scenarios...")
        with mp.get_context("spawn").Pool(use_cores_count) as pool:
            pool.map(run, [scenarios[i] for i in range(start_idx, min(use_cores_count+start_idx, end_idx))])


def run_scenarios_sequential():
    """Creates scenarios based on the energy and c-rate lists/vectors and runs each of the scenarios,
    which is a combination of all the capacities and c-rates"""
    start_idx = 0
    end_idx = 10
    idx_list = list(range(start_idx, end_idx, 1))
    # idx_list = [start_idx, end_idx]
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
                'index': idx}
    run(scenario)


if __name__ == '__main__':
    if sequential_run:
        run_scenarios_sequential()
    elif single_run:
        run_scenario_single()
    else:
        run_scenarios_parallel()
