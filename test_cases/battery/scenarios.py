"""
**Overview**

This file is mainly used to run scenarios in parallel (if enough CPUs) or sequentially,
without having to modify and rerun the simulation each time.
Currently, user defines the battery Capacities (Wh) they want to compare and the maximum allowable battery C-rates.
"""
import os
import sys
import numpy as np
import multiprocessing as mp
import ast
sys.path.append('../../charging_sim')
from utils import month_days


# RUN TYPE
sequential_run = True
parallel_run = False
single_run = False


def make_month_str(month_int: int):
    """
    Makes a month string from the month integer. Adds 0 if the month is less than 10.

    :param month_int: 1 - January, 2 - February, etc.
    :return: String of the month.
    """
    if month_int >= 10:
        return str(month_int)
    else:
        return f'0{str(month_int)}'


def load_input_config():
    """
    Loads the configuration file for the simulation and returns a dict

    :return:
    """
    import json
    with open('../../../EV50_cosimulation/user_input.json', 'r') as f:
        config = json.load(f)
    return config


def make_scenarios():
    """
    This is used to make the list of scenarios (dicts) that are used to run the simulations.
    No inputs. However, it uses preloaded global functions from a `config.txt` file based on the user
    settings and inputs. Please change JSON file directly for multiple scenarios.

    :return: List of scenario dicts.
    """
    inputs = load_input_config()
    # Preload.
    station_config = open('feeder_population/config.txt', 'r')
    param_dict = ast.literal_eval(station_config.read())
    station_config.close()
    start_time = param_dict['starttime'][:6] + make_month_str(inputs['month']) + param_dict['starttime'][8:]
    end_time = param_dict['endtime'][:6] + make_month_str(inputs['month']) + param_dict['endtime'][8:]

    charging_station_config = inputs["charging_station"]
    battery_config = inputs["battery"]
    solar_config = inputs["solar"]
    DAY_MINUTES = 1440
    OPT_TIME_RES = 15  # minutes
    NUM_DAYS = inputs["num_days"]  # determines optimization horizon
    NUM_STEPS = NUM_DAYS * DAY_MINUTES // OPT_TIME_RES  # number of steps to initialize variables for opt
    print("basic configs done...")

    # Modify configs based on user inputs.
    # Modify the config file in feeder population based on the user inputs.
    # Append to list of capacities as the user adds more scenarios. Limit the max user scenarios that can be added.

    # Modify param dict.
    param_dict['starttime'] = f'{start_time}'
    param_dict['endtime'] = f'{end_time}'

    # print(charging_station_config)
    # Control user inputs for charging stations.
    if charging_station_config["num_l2_stalls_per_station"] and charging_station_config["num_dcfc_stalls_per_station"]:
        raise ValueError("Cannot have both L2 and DCFC charging stations at the same time.")

    # Updating initial param dict with user inputs, new param dict will be written to the config.txt file.
    # print(charging_station_config)

    if charging_station_config['num_dcfc_stalls_per_station']:
        param_dict['num_dcfc_stalls_per_station'] = charging_station_config['num_dcfc_stalls_per_station']
        if charging_station_config["dcfc_charging_stall_base_rating"]:
            param_dict[
                'dcfc_charging_stall_base_rating'] = f'{charging_station_config["dcfc_charging_stall_base_rating"]}_kW'

    if charging_station_config['num_l2_stalls_per_station']:
        param_dict['num_l2_stalls_per_station'] = charging_station_config['num_l2_stalls_per_station']
        if charging_station_config["l2_power_cap"]:
            param_dict['l2_charging_stall_base_rating'] = f'{charging_station_config["l2_power_cap"]}_kW'

    # Obtaining the charging station capacities.
    dcfc_station_cap = float(param_dict['dcfc_charging_stall_base_rating'].split('_')[0]) * param_dict[
        'num_dcfc_stalls_per_station']
    L2_station_cap = float(param_dict['l2_charging_stall_base_rating'].split('_')[0]) * param_dict[
        'num_l2_stalls_per_station']
    month = int(str(param_dict['starttime']).split('-')[1])
    # Month index starting from 1. e.g. 1: January, 2: February, 3: March etc.
    month_str = list(month_days.keys())[month - 1]

    # Save the new param_dict to the config file.
    station_config = open('feeder_population/config.txt', 'w')
    station_config.writelines(', \n'.join(str(param_dict).split(',')))
    station_config.close()

    # Load DCFC locations txt file.
    print('...loading charging bus nodes')
    dcfc_nodes = np.loadtxt('dcfc_bus.txt', dtype=str).tolist()  # This is for DC FAST charging.
    if type(dcfc_nodes) is not list:
        dcfc_nodes = [dcfc_nodes]
    dcfc_dicts_list = []
    for node in dcfc_nodes:
        dcfc_dicts_list += {"DCFC": dcfc_station_cap, "L2": 0, "node_name": node},

    L2_charging_nodes = np.loadtxt('L2charging_bus.txt', dtype=str).tolist()  # this is for L2
    if type(L2_charging_nodes) is not list:
        L2_charging_nodes = [L2_charging_nodes]
    l2_dicts_list = []
    for node in L2_charging_nodes:
        l2_dicts_list += {"DCFC": 0, "L2": L2_station_cap, "node_name": node},
    num_charging_nodes = len(dcfc_nodes) + len(L2_charging_nodes)
    # Needs to come in as input initially & should be initialized prior from the feeder population.

    #   RUN TYPE - User may be able to choose parallel or sequential run. Will need to stress-test the parallel run.
    #   (Does not work currently)

    # Battery scenarios.
    energy_ratings = battery_config["pack_energy_cap"]  # kWh
    max_c_rates = battery_config["max_c_rate"]

    # New code starts here.
    scenarios_list = []
    voltage_idx, idx = 0, 0
    # Seems like we don't get list[int] for voltages
    for Er in energy_ratings:
        for c_rate in max_c_rates:
            scenario = {
                'index': idx,
                'oneshot': False,
                'start_month': month,
                'month_str': month_str,
                'opt_solver': 'GUROBI',
                'battery': {
                    'pack_energy_cap': Er,
                    'max_c_rate': c_rate,
                    'pack_max_voltage': inputs['battery']['pack_max_voltage'][voltage_idx]
                },
                'charging_station': {
                    'dcfc_power_cap': dcfc_station_cap
                },
                'solar': {
                    'start_month': month,
                    'efficiency': solar_config["efficiency"],
                    'rating': solar_config["rating"],
                    'data_path': solar_config["data"]
                },
                'load': {
                    'data_path': inputs['load']['data']
                },
                'elec_prices': {
                    'start_month': month,
                    'data_path': inputs['elec_prices']['data']
                }
            }
            scenarios_list.append(scenario)
            idx += 1
        voltage_idx += 1
    return scenarios_list


def run(scenario):
    """
    Runs a given scenario.

    :param dict scenario: Scenario dictionary containing inputs a user would like to run.
    :return: None.
    """
    import master_sim
    master_sim.run(scenario)


def run_scenarios_parallel():
    """
    This runs c-rate-energy scenarios in parallel, using the multicore processor of the PC.
    User should have enough cores and RAM, as if not enough, can lead to entire process freezing.
    """
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
    """
    Creates scenarios based on the energy and c-rate lists/vectors and runs each of the scenarios,
    which is a combination of all the capacities and c-rates.
    """
    start_idx = 0
    end_idx = 10
    scenarios_list = make_scenarios()
    idx_list = list(range(start_idx, min(len(scenarios_list), end_idx), 1))
    scenarios = [scenarios_list[idx] for idx in idx_list]
    for scenario in scenarios:
        process = mp.get_context('spawn').Process(target=run, args=(scenario,))
        process.start()
        process.join()


def run_scenario_single():
    """
    Runs only one scenario from a user specified configuration file.
    """
    # Keep changing this for each run
    scenarios_list = make_scenarios()
    Er_idx = 0
    c_rate_idx = 2
    idx = 2
    scenario = scenarios_list[0]
    run(scenario)


if __name__ == '__main__':
    if sequential_run:
        run_scenarios_sequential()
    elif single_run:
        run_scenario_single()
    else:
        run_scenarios_parallel()
