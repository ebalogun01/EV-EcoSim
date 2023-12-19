"""
**Overview**
This module is the main power-horse for the `EV-EcoSim`. It includes the modules that allow GridLabD to interact
with all the custom-built modules developed in `EV-EcoSim`. This file imports all simulated objects and their children,
which is then run in the co-simulation environment.
"""

import os
import numpy as np
import gridlabd
import time
import gblvar
import json
import pandas as pd
from orchestrator import ChargingSim
from transformer import OilTypeTransformer
from clock import Clock

path_prefix = str(os.getcwd())
os.chdir(path_prefix)  # change directory
# Splitting the path is different for Windows and Linux/MacOS. Need condition to deal with both OS file path styles.
if '\\' in path_prefix:
    path_prefix = "/".join(
        path_prefix.split('\\')[:-2])  # Gets absolute path to the root of the project to get the desired files.
else:
    path_prefix = "/".join(path_prefix.split('/')[:-2])

save_folder_prefix = f'{gblvar.scenario["month_str"]}{gblvar.scenario["index"]}/'  # how can I permanently save this state?

# SET OPTIMIZATION SOLVER
solver_options = ['GUROBI', 'MOSEK', 'ECOS']
gblvar.scenario['opt_solver'] = solver_options[0]

# lood DCFC locations txt file
print('...loading charging bus nodes')
dcfc_nodes = np.loadtxt('dcfc_bus.txt', dtype=str).tolist()  # this is for DC FAST charging

central_der = True  # Toggle for central vs. decentralized storage.
if central_der:
    with open('feeder_node_dict.json') as f:
        central_der_nodes_dict = json.load(f)   # Primary to secondary node_name dictionary for tracking all the DERs.
    with open('der_load_dict.json') as f:
        central_der_load_dict = json.load(f)
    central_der_nodes = list(central_der_nodes_dict.keys())  # List of all the primary nodes with DERs.

if type(dcfc_nodes) is not list:
    dcfc_nodes = [dcfc_nodes]
dcfc_dicts_list = [{"DCFC": gblvar.scenario['charging_station']['dcfc_power_cap'], "L2": 0, "node_name": node}
                   for node in dcfc_nodes]

L2_charging_nodes = np.loadtxt('L2charging_bus.txt', dtype=str).tolist()  # this is for L2 charging
if type(L2_charging_nodes) is not list:
    L2_charging_nodes = [L2_charging_nodes]
l2_dicts_list = []
for node in L2_charging_nodes:
    l2_dicts_list += {"DCFC": 0, "L2": gblvar.scenario['charging_station']['L2_power_cap'], "node_name": node},

gblvar.scenario['L2_nodes'] = L2_charging_nodes
gblvar.scenario['dcfc_nodes'] = dcfc_nodes
num_charging_nodes = len(dcfc_nodes) + len(
    L2_charging_nodes)  # needs to come in as input initially & should be initialized prior from the feeder population



# AMBIENT CONDITIONS FOR TRANSFORMER SIMULATION
simulation_month = gblvar.scenario[
    'start_month']  # Months are indexed starting from 1.
temperature_data = pd.read_csv('../../ambient_data/trans_ambientT_timeseries.csv')
temperature_data = temperature_data[temperature_data['Month'] == simulation_month]['Temperature'].values

global tic, toc  # used to time simulation
tic = time.time()
#####

# Load the clock from its JSON file and instantiate the global clock object (from charging_sim, NOT GridLAB-D).
with open('../../charging_sim/configs/clock.json') as f:
    clock_config = json.load(f)
global_clock = Clock(clock_config)

# Initialize Charging Simulation
if central_der:
    EV_charging_sim = ChargingSim(num_charging_nodes, path_prefix=path_prefix, month=simulation_month,
                                  centralized=central_der, central_der_dict=central_der_nodes_dict)
else:
    EV_charging_sim = ChargingSim(num_charging_nodes, path_prefix=path_prefix, month=simulation_month)

# Load the transformer configuration prototype JSON file.
with open('../../charging_sim/configs/transformer.json') as f:
    transformer_config = json.load(f)
trans_map = {}  # Dictionary of transformer objects.


def on_init(t):
    """
    This defines the actions to take at very beginning of simulation, like getting objects and properties from gridlabd.

    :param t: arbitrary placeholder.
    :return: True - arbitrary value.
    """
    # get object lists from GridLAB-D
    print("Gridlabd Init Begin...")
    gridlabd.output("timestamp,x")
    gridlabd.set_value("voltdump", "filename", f'{save_folder_prefix}volt_dump.csv')
    # gblvar.node_list = find("class=node_name") # Never used currently, so commented out.
    # gblvar.load_list = find("class=load")
    # gblvar.sim_file_path = save_folder_prefix
    # gblvar.tn_list = find("class=triplex_node")
    # gblvar.trans_list = find("class=transformer") # Get all transformers.
    gblvar.trans_list = find_subset_trans("class=transformer")  # Only get transformers connected to EV charging.
    # gblvar.transconfig_list = find("class=transformer_configuration")

    # NEED TO INCLUDE A PRE-LAYER FOR FEEDER POPULATION FOR A GIVEN SIMULATION - use makefile to do this.
    EV_charging_sim.setup(dcfc_dicts_list + l2_dicts_list, scenario=gblvar.scenario)
    print("Making results directory at: ", save_folder_prefix)
    if not os.path.isdir(save_folder_prefix):
        os.mkdir(save_folder_prefix)
    np.savetxt(f'{save_folder_prefix}voltdump.txt', np.array([save_folder_prefix]), fmt="%s")
    return True


def on_precommit(t):
    """
    This runs the simulation, propagating all the states of the necessary physical objects and the grid.

    :param t: Arbitrary placeholder.
    :return bool: True
    """
    # get clock from GridLAB-D
    clock = gridlabd.get_global("clock")
    print(f'****  {str(clock)}  ****')

    # get voltage from previous timestep from GridLAB-D
    vm_array, vp_array = get_voltage()

    # get nominal voltages if first timestep
    if global_clock.it == 0:
        gblvar.nom_vmag = vm_array

    # initialize voltage vector
    if global_clock.it == 1:
        gblvar.vm = vm_array.reshape(1, -1)
        gblvar.vp = vp_array.reshape(1, -1)

    # concatenate new voltage vector onto voltage history
    elif global_clock.it > 1:
        gblvar.vm = np.concatenate((gblvar.vm, vm_array.reshape(1, -1)), axis=0)
        gblvar.vp = np.concatenate((gblvar.vp, vp_array.reshape(1, -1)), axis=0)

    # get transformer ratings and possibly other properties if first timestep
    if global_clock.it == 0:
        """
        This is where the transformers get initially instantiated. Done only once.
        """
        gblvar.trans_loading_percent = []
        for i in range(len(gblvar.trans_list)):
            name = gblvar.trans_list[i]  # Gets one transformer from this list of transformers.
            data = gridlabd.get_object(name)  # USE THIS TO GET ANY OBJECT NEEDED
            trans_config_name = data['configuration']
            data = gridlabd.get_object(trans_config_name)
            power_rating = float(data['power_rating'].split(' ')[0])
            transformer_config['name'] = name
            transformer_config['rated-power'] = power_rating
            trans = OilTypeTransformer(transformer_config, global_clock=global_clock,
                                       temperature_data=temperature_data)
            trans_map[name] = trans  # Add the transformer to the transformer map.

    # get transformer power from previous timestep
    gblvar.trans_power = []
    #   Here we simulate the thermal dynamics of all transformers, however we can speed up by only simulated a subset
    for i in range(len(gblvar.trans_list)):
        name = gblvar.trans_list[i]
        data = gridlabd.get_object(name)
        trans_power_str = data['power_in']  # This gets us the power flowing through the transformer at the current time step.
        pmag, pdeg = get_trans_power(trans_power_str)
        trans_power_kVA = pmag / 1000  # in units kVA
        trans_map[name].thermal_dynamics(trans_power_kVA)  # Propagate the transformer state forward.


    ################################# CALCULATE POWER INJECTIONS FOR GRIDLABD ##########################################

    # calculate base_power and pf quantities to set for this timestep
    name_list_base_power = list(gblvar.p_df.columns)
    # set_power_vec = np.zeros((len(name_list_base_power),), dtype=complex)
    # for i in range(len(name_list_base_power)):
    #     # todo: is this part necessary? Seems like a bottleneck.
    #     set_power_vec[i] = gblvar.p_df[name_list_base_power[i]][global_clock.it] + gblvar.q_df[name_list_base_power[i]][
    #         global_clock.it] * 1j

    if global_clock.it % EV_charging_sim.resolution == 0:
        # Only step when controller time matches pf..based on resolution.
        # This ensures allows for varied resolution for ev-charging vs pf solver"""
        num_steps = 1
        if central_der:
            EV_charging_sim.step_centralized(num_steps)
        else:
            EV_charging_sim.step(num_steps)

    ################################## SEND TO GRIDLABD ################################################

    # Set base_power properties for this timestep, i.e. existing loads, home base loads, buildings, etc.
    # TODO: get regular L2 charging sharing transformer with existing loads
    prop = 'constant_power_12'
    for i in range(len(name_list_base_power)):
        name = name_list_base_power[i]
        total_node_load = 0
        # if ev node_name is power node_name, add ev_charging power to the set value for power vec (ONLY L2 CHARGING).
        if name in L2_charging_nodes:
            # This works because L2 Charging Nodes are modelled with existing triplex nodes.
            # We do not give L2 a separate transformer.
            charger = EV_charging_sim.get_charger_obj_by_loc(name)
            total_node_load += charger.get_current_load() * 1000  # for L2 (converting to Watts)
        base_power = gblvar.p_df[name_list_base_power[i]][global_clock.it] + gblvar.q_df[name_list_base_power[i]][
            global_clock.it] * 1j
        gridlabd.set_value(name, prop, str(base_power + total_node_load))

    # Set fast charging power properties for this timestep.
    prop_1 = 'constant_power_A'
    prop_2 = 'constant_power_B'
    prop_3 = 'constant_power_C'
    for name in dcfc_nodes:
        charger = EV_charging_sim.get_charger_obj_by_loc(name)
        total_node_load_watts = charger.get_current_load() * 1000  # converting to watts
        gridlabd.set_value(name, prop_1, str(total_node_load_watts / 3))  # balancing dcfc load between 3-phase
        gridlabd.set_value(name, prop_2, str(total_node_load_watts / 3))
        gridlabd.set_value(name, prop_3, str(total_node_load_watts / 3))

    # Centralized storage discharge/charge compensation (Might need to include inverter/transformer for voltage drop)
    # Currently assuming perfect inverter with 3-phase connection.
    if central_der:
        prop_1 = 'constant_power_A'
        prop_2 = 'constant_power_B'
        prop_3 = 'constant_power_C'
        for name in central_der_nodes:
            central_der_node = EV_charging_sim.central_node_dict[name]
            total_node_load_watts = central_der_node.get_current_load() * 1000  # converting to watts
            gridlabd.set_value(central_der_load_dict[name], prop_1, str(total_node_load_watts / 3))  # balancing dcfc load between 3-phase
            gridlabd.set_value(central_der_load_dict[name], prop_2, str(total_node_load_watts / 3))
            gridlabd.set_value(central_der_load_dict[name], prop_3, str(total_node_load_watts / 3))

    # increment timestep
    gblvar.it = gblvar.it + 1
    global_clock.update()
    return True


def on_term(t):
    """
    Actions taken at the very end of the whole simulation, like saving data.
    Data not required can be commented out on the relevant line.

    :param t: Default placeholder (do not change).
    :return: True.
    """
    EV_charging_sim.load_results_summary(save_folder_prefix)
    # np.savetxt(f'{save_folder_prefix}volt_mag.txt', gblvar.vm)
    # np.savetxt(f'{save_folder_prefix}volt_phase.txt', gblvar.vp)
    np.savetxt(f'{save_folder_prefix}nom_vmag.txt', gblvar.nom_vmag)  # nominal voltage magnitude (use in analysis)
    save_transformer_states()
    with open(f'{save_folder_prefix}scenario.json', "w") as outfile:
        json.dump(gblvar.scenario, outfile, indent=1)

    import voltdump2
    voltdump2.parse_voltages(save_folder_prefix)
    print("Total run time: ", (time.time() - tic) / 60, "minutes")
    return True


def find(criteria: str):
    """
    Finds and returns objects in GridLAB-D that satisfy certain criteria.

    :param str criteria: the criterion for returning gridlabd objects e.g. node_name, load, etc.
    :return: list of objects that satisfy the criteria.
    """
    finder = criteria.split("=")
    if len(finder) < 2:
        raise IOError("find(criteria='key=value'): criteria syntax error")
    objects = gridlabd.get("objects")
    result = []
    for name in objects:
        item = gridlabd.get_object(name)
        if finder[0] in item and item[finder[0]] == finder[1]:
            if "name" in item.keys():
                result.append(item["name"])
            else:
                result.append(f'{item["class"]}:{item["id"]}')  # Not sure I understand this line.
    return result


def find_subset_trans(criteria: str):
    """
    Finds and returns objects in gridlabd that satisfy certain criteria. This function only gets transformers
    for which are connected to EV charging, to save compute time.

    :param str criteria: the criterion for returning gridlabd objects e.g. node_name, load, etc.
    :return: list of objects that satisfy the criteria.
    """
    finder = criteria.split("=")
    if len(finder) < 2:
        raise IOError("find(criteria='key=value'): criteria syntax error")
    objects = gridlabd.get("objects")
    result = []
    for name in objects:
        item = gridlabd.get_object(name)
        if finder[0] in item and item[finder[0]] == finder[1]:
            if "name" in item.keys() and ("dcfc" in item["name"] or "L2" in item["name"]):
                result.append(item["name"])
    return result


def save_transformer_states():
    """
    Saves the relevant transformer states to a CSV file.

    :return:
    """
    first = True
    relevant_trans_keys = [key for key in trans_map.keys() if 'dcfc' in key]
    for trans_name in relevant_trans_keys:
        if first:
            trans_map[trans_name].plot_states()
            trans_Th = np.array(trans_map[trans_name].Th_list).reshape(-1, 1)
            trans_To = np.array(trans_map[trans_name].To_list).reshape(-1, 1)
            trans_loading = np.array(trans_map[trans_name].loading_percents).reshape(-1, 1)
            first = False
        else:
            trans_Th = np.hstack((trans_Th, np.array(trans_map[trans_name].Th_list).reshape(-1, 1)))
            trans_To = np.hstack((trans_To, np.array(trans_map[trans_name].To_list).reshape(-1, 1)))
            trans_loading = np.hstack((trans_loading, np.array(trans_map[trans_name].loading_percents).reshape(-1, 1)))
    # return
    pd.DataFrame(data=trans_Th, columns=relevant_trans_keys).to_csv(
        f'{save_folder_prefix}/trans_Th.csv', index=False)
    pd.DataFrame(data=trans_To, columns=relevant_trans_keys).to_csv(
        f'{save_folder_prefix}/trans_To.csv', index=False)
    pd.DataFrame(data=trans_loading, columns=relevant_trans_keys).to_csv(
        f'{save_folder_prefix}/trans_loading_percent.csv', index=False)


def get_voltage():
    """
    Obtains voltage string from GridLAB-D and processes it into float. GridLABD returns voltages in various
    formats, thus this processes the string voltages from the different formats into float. For more
    information on the formats, see the GridLabD powerflow user guide.

    :return: Processed voltage magnitude and voltage phase arrays.
    :rtype: ndarray(float).
    """
    vm_array = np.zeros((len(gblvar.voltage_obj),))
    vp_array = np.zeros((len(gblvar.voltage_prop),))
    for i in range(len(gblvar.voltage_obj)):
        name = gblvar.voltage_obj[i]
        prop = gblvar.voltage_prop[i]
        data = gridlabd.get_object(name)
        if 'e-' in data[prop]:
            if 'd' in data[prop]:
                data[prop] = data[prop].replace('e-', '(')
                vl = data[prop].rstrip('d V').replace('+', ',+').replace('-', ',-').split(',')
                if '(' in vl[1]:
                    vl[1] = vl[1].replace('(', 'e-')
                if '(' in vl[2]:
                    vl[2] = vl[2].replace('(', 'e-')
                vm_array[i] = float(vl[1])
                vp_array[i] = float(vl[2])
            elif 'j' in data[prop]:
                data[prop] = data[prop].replace('e-', '(')
                vl = data[prop].rstrip('j V').replace('+', ',+').replace('-', ',-').split(',')
                if '(' in vl[1]:
                    vl[1] = vl[1].replace('(', 'e-')
                if '(' in vl[2]:
                    vl[2] = vl[2].replace('(', 'e-')
                vm_array[i] = (float(vl[1]) ** 2 + float(vl[2]) ** 2) ** 0.5
                vp_array[i] = np.rad2deg(np.angle(float(vl[1]) + float(vl[2]) * 1j))

            else:
                # print(data[prop])  # removed x with throws error
                raise IOError('Missing string in transformer power string')

        elif 'd' in data[prop]:
            vl = data[prop].rstrip('d V').replace('+', ',+').replace('-', ',-').split(',')
            vm_array[i] = float(vl[1])
            vp_array[i] = float(vl[2])

        else:
            # think the fix ideally should be here but doesn't change time complexity
            vl = data[prop].rstrip('j V').replace('+', ',+').replace('-', ',-').split(',')
            vm_array[i] = (float(vl[1]) ** 2 + float(vl[2]) ** 2) ** 0.5
            vp_array[i] = np.rad2deg(np.angle(float(vl[1]) + float(vl[2]) * 1j))
    return vm_array, vp_array


def get_trans_power(trans_power_str):
    """
    Obtains power (in kVA) at transformer as a string and processes it into a float.

    :param trans_power_str: Transformer power as a string.
    :return pmag: Power magnitude.
    :return deg: Angle between pmag (apparent power) and the real axis on the complex power plane.
    """
    trans_power_str = trans_power_str.rstrip(' VA')
    if 'e-' in trans_power_str:
        if 'd' not in trans_power_str:
            # print(trans_power_str)  # removed x with throws error
            raise IOError('Missing string in transformer power string')

        trans_power_str = trans_power_str.replace('e-', '(')
        strtemp = trans_power_str.rstrip('d').replace('+', ',+').replace('-', ',-').split(',')
        if '(' in strtemp[1]:
            strtemp[1] = strtemp[1].replace('(', 'e-')
        if '(' in strtemp[2]:
            strtemp[2] = strtemp[2].replace('(', 'e-')
        pmag = float(strtemp[1])
        pdeg = float(strtemp[2])
    elif 'd' in trans_power_str:
        strtemp = trans_power_str.rstrip('d').replace('+', ',+').replace('-', ',-').split(',')
        pmag = float(strtemp[1])
        pdeg = float(strtemp[2])

    else:
        strtemp = trans_power_str.rstrip('j').replace('+', ',+').replace('-', ',-').split(',')

        pmag = (float(strtemp[1]) ** 2 + float(strtemp[2]) ** 2) ** 0.5
        pdeg = np.rad2deg(np.angle(float(strtemp[1]) + float(strtemp[2]) * 1j))

    return pmag, pdeg
