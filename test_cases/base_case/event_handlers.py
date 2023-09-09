"""This module is the main power-horse for the EV-Ecosim. It includes the modules that allow GridLabD to interact
with all the custom-built modules developed in EV-Ecosim. This file imports all simulated objects and their children,
which is then run in the co-simulation environment."""

import os
import numpy as np
import gridlabd
import pandas as pd
import time
import gblvar
import sim
from orchestrator import ChargingSim

#   will later remove some import flags but leaving here for potential debugging

# get the desired path prefix
path_prefix = os.getcwd()
path_prefix = path_prefix[0:path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
path_prefix.replace('\\', '/')
save_folder_prefix = 'June_test/'
num_charging_nodes = 0  # needs to come in as input initially & should be initialized prior from the feeder population
central_storage = False  # toggle for central vs. decentralized storage

# AMBIENT CONDITIONS FOR TRANSFORMER SIMULATION
simulation_month = 6  # Months are indexed starting from 1 - CHANGE MONTH (TO BE AUTOMATED LATER)
temperature_data = pd.read_csv('../../ambient_data/trans_ambientT_timeseries.csv')
temperature_data = temperature_data[temperature_data['Month'] == simulation_month]['Temperature'].values


# Sets up the simulation module with Charging sites and batteries
global tic
tic = time.time()

EV_charging_sim = ChargingSim(num_charging_nodes, path_prefix=path_prefix)  # Initialize Charging Simulation


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
    gblvar.node_list = find("class=node")
    gblvar.load_list = find("class=load")
    gblvar.tn_list = find("class=triplex_node")
    gblvar.trans_list = find("class=transformer")
    gblvar.transconfig_list = find("class=transformer_configuration")

    # NEED TO maybe INCLUDE A PRE-LAYER FOR FEEDER POPULATION FOR A GIVEN SIMULATION
    EV_charging_sim.setup(list(gblvar.tn_list))
    print("Making results directory at: ", save_folder_prefix)
    os.mkdir(save_folder_prefix)
    np.savetxt(f'{save_folder_prefix}voltdump.txt', np.array([save_folder_prefix]), fmt="%s")
    return True


def on_precommit(t):
    """
    This runs the simulation, propagating all the states of the necessary physical objects and the grid.

    :param t: Arbitrary placeholder.
    :return: True
    """
    clock = gridlabd.get_global("clock")
    print(f'****  {str(clock)}  ****')

    # get voltage from GridLAB-D
    vm_array, vp_array = get_voltage()

    # get nominal voltages if first timestep
    if gblvar.it == 0:
        gblvar.nom_vmag = vm_array

    # initialize voltage vector
    if gblvar.it == 1:
        gblvar.vm = vm_array.reshape(1, -1)
        gblvar.vp = vp_array.reshape(1, -1)
    # concatenate new voltage vector onto voltage history
    elif gblvar.it > 1:
        gblvar.vm = np.concatenate((gblvar.vm, vm_array.reshape(1, -1)), axis=0)
        gblvar.vp = np.concatenate((gblvar.vp, vp_array.reshape(1, -1)), axis=0)
    # print(vm_array[-1])

    # get transformer ratings and possibly other properties if first timestep
    if gblvar.it == 0:
        gblvar.trans_rated_s = []
        gblvar.trans_loading_percent = []
        for i in range(len(gblvar.trans_list)):
            name = gblvar.trans_list[i]
            data = gridlabd.get_object(name)  # USE THIS TO GET ANY OBJECT NEEDED
            trans_config_name = data['configuration']
            data = gridlabd.get_object(trans_config_name)
            gblvar.trans_rated_s.append(float(data['power_rating'].split(' ')[0]))
        gblvar.trans_rated_s_np = np.array(gblvar.trans_rated_s).reshape(1, -1)

    # get transformer power from previous timestep
    gblvar.trans_power = []
    for i in range(len(gblvar.trans_list)):
        name = gblvar.trans_list[i]
        data = gridlabd.get_object(name)
        trans_power_str = data['power_in']
        # print(trans_power_str)
        pmag, pdeg = get_trans_power(trans_power_str)
        gblvar.trans_power.append(pmag / 1000)  # in units kVA
    if gblvar.it == 0:
        gblvar.trans_loading_percent = np.array(gblvar.trans_power).reshape(1, -1) / gblvar.trans_rated_s_np
    else:
        gblvar.trans_loading_percent = np.vstack((gblvar.trans_loading_percent,
                                                  np.array(gblvar.trans_power).reshape(1, -1) / gblvar.trans_rated_s_np))   # done

    ####################### SIMULATE ##################################
    # propagate transformer state
    sim.sim_transformer(temperature_data)

    ################################# CALCULATE POWER INJECTIONS FOR GRIDLABD ##########################################

    # calculate base_power and pf quantities to set for this timestep
    name_list_base_power = list(gblvar.p_df.columns)
    set_power_vec = np.zeros((len(name_list_base_power),), dtype=complex)
    # print("Global time is: ", gblvar.it)
    if num_charging_nodes and gblvar.it % EV_charging_sim.resolution == 0:
        """only step when controller time matches pf..based on resolution.
        This ensures varying resolution for ev-charging vs pf solver"""
        # get loads from EV charging station
        num_steps = 1
        EV_charging_sim.step(num_steps)  # get loads from EV charging station

    for i in range(len(name_list_base_power)):  # add EV simulation net load for each location
        set_power_vec[i] = gblvar.p_df[name_list_base_power[i]][gblvar.it] + gblvar.q_df[name_list_base_power[i]][
            gblvar.it] * 1j

    # set base_power properties for this timestep
    charging_nodes = EV_charging_sim.get_charging_sites()
    if central_storage:
        central_storage_nodes = EV_charging_sim.get_storage_sites()
    prop = 'power_12'  # power/load
    for i in range(len(name_list_base_power)):
        # here
        name = name_list_base_power[i]
        node_ev_load = 0
        # if ev node is power node, add ev_charging power to the set value for power vec.
        if name in charging_nodes:
            charger = EV_charging_sim.get_charger_obj_by_loc(name)
            node_ev_load += charger.get_current_load()
        if central_storage and name in central_storage_nodes:
            storage = EV_charging_sim.get_storage_obj_by_loc(name)
            storage_load = storage.power  # units in kW (should be negative if there is discharge to the grid/charger)
            node_ev_load += storage_load
        gridlabd.set_value(name, prop, str(set_power_vec[i] + node_ev_load).replace('(', '').replace(')', ''))

    # increment timestep
    gblvar.it = gblvar.it + 1
    return True


def on_term(t):
    """
    Actions taken at the very end of the whole simulation, like saving data.
    Data not required can be commented out on the relevant line.

    :param t: Default placeholder (do not change).
    :return: True.
    """
    import voltdump2
    if num_charging_nodes:
        EV_charging_sim.load_results_summary(save_folder_prefix)
    # np.savetxt(f'{save_folder_prefix}volt_mag.txt', gblvar.vm)
    # np.savetxt(f'{save_folder_prefix}volt_phase.txt', gblvar.vp)
    np.savetxt(f'{save_folder_prefix}nom_vmag.txt', gblvar.nom_vmag)  # nominal voltage magnitude (use in analysis)
    pd.DataFrame(data=gblvar.trans_Th, columns=gblvar.trans_list).to_csv(f'{save_folder_prefix}/trans_Th.csv',
                                                                         index=False)
    # pd.DataFrame(data=gblvar.trans_To, columns=gblvar.trans_list).to_csv(f'{save_folder_prefix}/trans_To.csv',
    #                                                                      index=False)
    pd.DataFrame(data=gblvar.trans_loading_percent, columns=gblvar.trans_list). \
        to_csv(f'{save_folder_prefix}/trans_loading_percent.csv',
               index=False)  # included saving transformer loading percentages
    voltdump2.parse_voltages(save_folder_prefix)
    print("Total run time: ", (time.time() - tic) / 60, "minutes")


def find(criteria):
    """
    Finds and returns objects in gridlabd that satisfy certain criteria.

    :param str criteria: the criterion for returning gridlabd objects e.g. node, load, etc.
    :return: list of objects that satisfy the criteria.
    """
    finder = criteria.split("=")
    if len(finder) < 2:
        raise SyntaxError("find(criteria='key=value'): criteria syntax error")
    objects = gridlabd.get("objects")
    result = []
    for name in objects:
        item = gridlabd.get_object(name)
        if finder[0] in item and item[finder[0]] == finder[1]:
            if "name" in item.keys():
                result.append(item["name"])
            else:
                result.append(f'{item["class"]}:{item["id"]}')
    return result


def get_voltage():
    """
    Obtains voltage string from GridLAB-D and processes it into float. GridLABD returns voltages in various
    formats, thus this processes the string voltages from the different formats into float. For more
    information on the formats, see the GridLabD powerflow user guide.

    :return: Processed voltage magnitude and voltage phase arrays.
    :rtype: ndarray(float).
    """
    #   TODO: find a way to ignore the nodes that have no voltage (zero-load)
    vm_array, vp_array = np.zeros((len(gblvar.voltage_obj),)), np.zeros((len(gblvar.voltage_prop),))
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
    Obtains power at transformer as a string and processes it into a float.

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
