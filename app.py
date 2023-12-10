"""
Runs the EV-EcoSim application. This is the main file that is run to start the application.
This module runs the optimization offline without the power system or battery state feedback for each time-step.
This is done to save time. Once this is done, one can study the effects on the power system. Power system states are
propagated post optimization to fully characterize what would have occurred if in-situ optimization was done.
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


def create_results_folder():
    """
    Creates a results dir if one does not exist

    :return:
    """
    if os.path.isdir('analysis/results'):
        return
    os.mkdir('analysis/results')


def load_default_input():
    """
    Loads the default user input skeleton.

    :return:
    """
    with open('default_user_input.json', "r") as f:
        user_input = json.load(f)
    validate_options(user_input)  # todo: Finish implementing this part later.
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
# user_inputs = load_default_input()


def simulate(user_inputs, sequential_run=True, parallel_run=False):
    # Updating the user inputs based on frontend inputs.
    create_results_folder()  # Make a results folder if it does not exist.

    path_prefix = os.getcwd()
    # Change below to name of the repo.
    results_folder_path = path_prefix[: path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation/analysis/results'
    path_prefix = path_prefix[: path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'

    # PRELOAD
    station_config = open(path_prefix + '/test_cases/battery/feeder_population/config.txt', 'r')
    param_dict = ast.literal_eval(station_config.read())
    station_config.close()
    start_time = param_dict['starttime'][:6] + make_month_str(user_inputs['month']) + param_dict['starttime'][8:]
    end_time = param_dict['endtime'][:6] + make_month_str(user_inputs['month']) + param_dict['endtime'][8:]

    charging_station_config = user_inputs["charging_station"]
    battery_config = user_inputs["battery"]
    solar_config = user_inputs["solar"]
    DAY_MINUTES = 1440
    OPT_TIME_RES = 15  # minutes
    NUM_DAYS = user_inputs["num_days"]  # determines optimization horizon
    NUM_STEPS = NUM_DAYS * DAY_MINUTES // OPT_TIME_RES  # number of steps to initialize variables for opt
    print("basic configs done...")

    # Modify configs based on user inputs.
    # Modify the config file in feeder population based on the user inputs.
    # Append to list of capacities as the user adds more scenarios. Limit the max user scenarios that can be added.

    # Modify param dict.
    param_dict['starttime'] = f'{start_time}'
    param_dict['endtime'] = f'{end_time}'

    print(charging_station_config)
    # Control user inputs for charging stations.
    if charging_station_config["num_l2_stalls_per_station"] and charging_station_config["num_dcfc_stalls_per_station"]:
        raise ValueError("Cannot have both L2 and DCFC charging stations at the same time.")

    # Updating initial param dict with user inputs, new param dict will be written to the config.txt file.
    print(charging_station_config)

    if charging_station_config['num_dcfc_stalls_per_station']:
        param_dict['num_dcfc_stalls_per_station'] = charging_station_config['num_dcfc_stalls_per_station']
        if charging_station_config["dcfc_charging_stall_base_rating"]:
            param_dict[
                'dcfc_charging_stall_base_rating'] = f'{charging_station_config["dcfc_charging_stall_base_rating"]}'

    if charging_station_config['num_l2_stalls_per_station']:
        param_dict['num_l2_stalls_per_node'] = charging_station_config['num_l2_stalls_per_node']
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
    station_config = open(path_prefix + '/test_cases/battery/feeder_population/config.txt', 'w')
    station_config.writelines(', \n'.join(str(param_dict).split(',')))
    station_config.close()

    # Load DCFC locations txt file.
    print('...loading charging bus nodes')
    dcfc_nodes = np.loadtxt('test_cases/battery/dcfc_bus.txt', dtype=str).tolist()  # This is for DC FAST charging.
    if type(dcfc_nodes) is not list:
        dcfc_nodes = [dcfc_nodes]
    dcfc_dicts_list = []
    for node in dcfc_nodes:
        dcfc_dicts_list += {"DCFC": dcfc_station_cap, "L2": 0, "node_name": node},

    L2_charging_nodes = np.loadtxt('test_cases/battery/L2charging_bus.txt', dtype=str).tolist()  # this is for L2
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
    max_c_rates = battery_config["max_c_rate"]  # kW

    def make_scenarios():
        """
        Function to make the list of scenarios (dicts) that are used to run the simulations. Each scenario is fully
        specified by a dict. The dict produced is used by the orchestrator to run the simulation.

        :return: List of scenario dicts.
        """
        scenarios_list = []
        voltage_idx, idx = 0, 0
        # Seems like we don't get list[int] for voltages
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
                        'pack_max_voltage': user_inputs['battery']['pack_max_voltage'][voltage_idx]
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
                        'data_path': user_inputs['load']['data']
                    },
                    'elec_prices': {
                        'start_month': month,
                        'data_path': user_inputs['elec_prices']['data']
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
        save_folder_prefix = f'{results_folder_path}/oneshot_{month_str}{str(scenario["index"])}/'
        if not os.path.exists(save_folder_prefix):
            os.mkdir(save_folder_prefix)
        EV_charging_sim.setup(dcfc_dicts_list + l2_dicts_list, scenario=scenario)
        print('multistep')
        EV_charging_sim.multistep()
        print('multistep done')
        EV_charging_sim.load_results_summary(save_folder_prefix)
        with open(f'{save_folder_prefix}scenario.json', "w") as outfile:
            json.dump(scenario, outfile, indent=1)

    def run_scenarios_sequential():
        """
        Creates scenarios based on the energy and c-rate lists/vectors and runs each of the scenarios,
        which is a combination of all the capacities and c-rates.

        :return: None.
        """
        start_idx = 0
        end_idx = len(energy_ratings) * len(max_c_rates)
        idx_list = list(range(start_idx, end_idx))
        print('making scenarios')
        scenarios_list = make_scenarios()
        scenarios = [scenarios_list[idx] for idx in idx_list]
        for scenario in scenarios:
            print(scenario)
            scenario["L2_nodes"] = L2_charging_nodes
            scenario["dcfc_nodes"] = dcfc_nodes
            if dcfc_dicts_list:
                scenario["dcfc_caps"] = [station["DCFC"] for station in dcfc_dicts_list]
            if l2_dicts_list:
                scenario["l2_caps"] = [station["L2"] for station in l2_dicts_list]
            run(scenario)

    if sequential_run:
        print("Running scenarios sequentially...")
        run_scenarios_sequential()
        print("Simulation complete!")

    return


if __name__ == '__main__':
    USER_INPUTS = load_default_input()
    simulate(USER_INPUTS)
