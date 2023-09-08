"""
Runs the application from front-end to back-end.
"""
import sys
sys.path.append('./charging_sim')
import os
from charging_sim.orchestrator import ChargingSim
import multiprocessing as mp
import numpy as np
import json
import ast
from charging_sim.utils import month_days


# if running app and uploading data, the system must load the different classes with the right temp datasets.


def create_temp_configs():
    """
    Creates temporary configuration files used by the simulator for running the app.

    :return:
    """


def validate_options(front_input: dict):
    """
    Validates the user-input options to ensure that the options selected matches the required workflow by the backend.

    :return: None.
    """
    print("Validating user input options...")
    return None


def load_default_input():
    """
    Loads the default user input skeleton.

    :return:
    """
    with open('user_input.json', "r") as f:
        user_input = json.load(f)
    validate_options(user_input)    # todo: Finish implementing this part later.
    return user_input


def change_run_date():
    """
    Changes the run date for the simulation.

    :return:
    """
    pass


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

# GET THE PATH PREFIX FOR SAVING THE INPUTS


USER_INPUTS = load_default_input()
# Updating the user inputs based on frontend inputs.

path_prefix = os.getcwd()
# Change below to name of the repo.
path_prefix = path_prefix[: path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'


# PRELOAD
station_config = open(path_prefix + '/test_cases/battery/feeder_population/config.txt', 'r')
param_dict = station_config.read()
station_config.close()
param_dict = ast.literal_eval(param_dict)
start_time = param_dict['starttime'][:6] + make_month_str(USER_INPUTS['month']) + param_dict['starttime'][8:]
end_time = param_dict['endtime'][:6] + make_month_str(USER_INPUTS['month']) + param_dict['endtime'][8:]


charging_station_config = USER_INPUTS["charging_station"]
battery_config = USER_INPUTS["battery"]
solar_config = USER_INPUTS["solar"]
DAY_MINUTES = 1440
OPT_TIME_RES = 15  # minutes
NUM_DAYS = USER_INPUTS["num_days"]  # determines optimization horizon
NUM_STEPS = NUM_DAYS * DAY_MINUTES // OPT_TIME_RES  # number of steps to initialize variables for opt


# Modify configs based on user inputs.
# Modify the config file in feeder population based on the user inputs.
# Append to list of capacities as the user adds more scenarios. Limit the max user scenarios that can be added.


# Modify param dict.
param_dict['starttime'] = f'{start_time}'
param_dict['endtime'] = f'{end_time}'

if charging_station_config["num_l2_stalls_per_node"] and charging_station_config["num_dcfc_stalls_per_node"]:
    raise ValueError("Cannot have both L2 and DCFC charging stations at the same time.")

if charging_station_config['num_dcfc_stalls_per_node']:
    param_dict['num_dcfc_stalls_per_node'] = charging_station_config['num_dcfc_stalls_per_node']
    if charging_station_config["dcfc_power_cap"]:
        param_dict['dcfc_charging_stall_base_rating'] = f'{charging_station_config["dcfc_power_cap"]}_kW'

if charging_station_config['num_l2_stalls_per_node']:
    param_dict['num_l2_stalls_per_node'] = charging_station_config['num_l2_stalls_per_node']
    if charging_station_config["l2_power_cap"]:
        param_dict['l2_charging_stall_base_rating'] = f'{charging_station_config["l2_power_cap"]}_kW'


dcfc_station_cap = float(param_dict['dcfc_charging_stall_base_rating'].split('_')[0]) * \
                   param_dict['num_dcfc_stalls_per_node']
L2_station_cap = float(param_dict['l2_charging_stall_base_rating'].split('_')[0]) * param_dict['num_l2_stalls_per_node']
month = int(str(param_dict['starttime']).split('-')[1])
# Month index starting from 1. e.g. 1: January, 2: February, 3: March etc.
month_str = list(month_days.keys())[month - 1]

# Save the new param_dict to the config file.
station_config = open(path_prefix + '/test_cases/battery/feeder_population/config.txt', 'w')
station_config.write(str(param_dict))
station_config.close()

# Load DCFC locations txt file.
print('...loading charging bus nodes')
dcfc_nodes = np.loadtxt('test_cases/battery/dcfc_bus.txt', dtype=str).tolist()  # This is for DC FAST charging.
if type(dcfc_nodes) is not list:
    dcfc_nodes = [dcfc_nodes]
dcfc_dicts_list = []
for node in dcfc_nodes:
    dcfc_dicts_list += {"DCFC": dcfc_station_cap, "L2": 0, "node": node},

L2_charging_nodes = np.loadtxt('test_cases/battery/L2charging_bus.txt',
                               dtype=str).tolist()  # this is for L2 charging
if type(L2_charging_nodes) is not list:
    L2_charging_nodes = [L2_charging_nodes]
l2_dicts_list = []
for node in L2_charging_nodes:
    l2_dicts_list += {"DCFC": 0, "L2": L2_station_cap, "node": node},
num_charging_nodes = len(dcfc_nodes) + len(L2_charging_nodes)
# Needs to come in as input initially & should be initialized prior from the feeder population.

#   RUN TYPE
sequential_run = True
parallel_run = False

# BATTERY SCENARIOS
energy_ratings = USER_INPUTS["battery"]["pack_energy_cap"]  # kWh
max_c_rates = USER_INPUTS["battery"]["max_c_rate"]  # kW


def make_scenarios():
    """
    This is used to make the list of scenarios (dicts) that are used to run the simulations.
    No inputs. However, it uses preloaded global functions from a `config.txt` file.

    :return list scenarios_list: List of scenario dicts.
    """
    scenarios_list = []
    voltage_idx, idx = 0, 0
    for Er in energy_ratings:
        for c_rate in max_c_rates:
            scenario = {
                'index': idx,
                'oneshot': True,
                'start_month': month,
                'opt_solver': 'GUROBI',
                'battery': {
                    'pack_energy_cap': Er,
                    'max_c_rate': c_rate,
                    'pack_max_voltage': USER_INPUTS['battery']['pack_max_voltage'][voltage_idx]
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
                    'data_path': USER_INPUTS['load']['data']
                },
                'elec_prices': {
                    'start_month': month,
                    'data_path': USER_INPUTS['elec_prices']['data']
                }
            }
            scenarios_list.append(scenario)
            idx += 1
        voltage_idx += 1
    return scenarios_list


def run(scenario):
    """
    Runs a scenario and updates the scenario JSON to reflect main properties of that scenario.

    :param scenario: The scenario dictionary that would be run.
    :return: None. Runs the `scenario`.
    """
    EV_charging_sim = ChargingSim(num_charging_nodes, path_prefix=path_prefix, num_steps=NUM_STEPS, month=month)
    save_folder_prefix = f'oneshot_{month_str}{str(scenario["index"])}/'
    if not os.path.exists(save_folder_prefix):
        os.mkdir(save_folder_prefix)
    EV_charging_sim.setup(dcfc_dicts_list+l2_dicts_list, scenario=scenario)
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
    for scenario in scenarios:
        scenario["L2_nodes"] = L2_charging_nodes
        scenario["dcfc_nodes"] = dcfc_nodes
        if dcfc_dicts_list:
            scenario["dcfc_caps"] = [station["DCFC"] for station in dcfc_dicts_list]
        if l2_dicts_list:
            scenario["l2_caps"] = [station["L2"] for station in l2_dicts_list]
        run(scenario)


if __name__ == '__main__':
    if sequential_run:
        run_scenarios_sequential()
    else:
        run_scenarios_parallel()
