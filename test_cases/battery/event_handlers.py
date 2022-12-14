import sys
import os
import numpy as np
# import pandas as pd
import gridlabd
print("gridlab-D imported")
import sim
import time
import gblvar
import json

if not gblvar.charging_sim_path_append:
    sys.path.append('../../../EV50_cosimulation/charging_sim')    # change this
    print('append 2')
from EVCharging import ChargingSim
print("*****EV Charging Station Simulation Imported Successfully*****")

# get the desired path prefix


path_prefix = os.getcwd()
path_prefix = path_prefix[0:path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
path_prefix.replace('\\', '/')
save_folder_prefix = 'sim_'+str(gblvar.scenario['index'])   # how can I permanently save this state?

# lood DCFC locations txt file
print('...loading dcfc bus nodes')
dcfc_nodes = np.loadtxt('dcfc_bus.txt', dtype=str).tolist()
print("DCFC bus nodes loaded...")
num_charging_nodes = len(dcfc_nodes)  # needs to come in as input initially & should be initialized prior from the feeder population
central_storage = False     # toggle for central vs. decentralized storage
#####
global tic, toc     # used to time simulation
tic = time.time()
#####

EV_charging_sim = ChargingSim(num_charging_nodes, path_prefix=path_prefix)  # Initialize Charging Simulation


def on_init(t):
    '''Stuff to do at very beginning of simulation, like getting objects and properties from gridlabd'''
    # get object lists from GridLAB-D
    gridlabd.set_value("voltdump", "filename", save_folder_prefix)
    print("Gridlabd Init Begin...")
    gridlabd.output("timestamp,x")
    np.savetxt(path_prefix+'/voltdump.txt', np.array([save_folder_prefix + '/']), fmt="%s")
    gridlabd.set_value("voltdump", "filename", save_folder_prefix + "/" + 'volt_dump.csv')
    gblvar.node_list = find("class=node")
    gblvar.load_list = find("class=load")
    gblvar.tn_list = find("class=triplex_node")
    gblvar.trans_list = find("class=transformer")
    gblvar.transconfig_list = find("class=transformer_configuration")

    # Configure EV charging simulation...NEED TO INCLUDE A PRE-LAYER FOR FEEDER POPULATION FOR A GIVEN SIMULATION
    EV_charging_sim.setup(dcfc_nodes,  scenario=gblvar.scenario)
    print("Making results directory at: ", save_folder_prefix)
    os.mkdir(save_folder_prefix)
    return True


def on_precommit(t):

    ########################## UPDATES FROM GRIDLABD ##################################

    # get clock from GridLAB-D
    clock = gridlabd.get_global("clock")
    print('****  ' + str(clock) + '  ****')

    # get voltage from previous timestep from GridLAB-D
    vm_array, vp_array = get_voltage()

    # get nominal voltages if first timestep
    if gblvar.it == 0:
        gblvar.nom_vmag = vm_array

    # initialize voltage vector
    if gblvar.it == 1:
        gblvar.vm = vm_array.reshape(1, -1)
        gblvar.vp = vp_array.reshape(1, -1)

    # concatenace new voltage vector onto voltage history
    elif gblvar.it > 1:
        gblvar.vm = np.concatenate((gblvar.vm, vm_array.reshape(1, -1)), axis=0)
        gblvar.vp = np.concatenate((gblvar.vp, vp_array.reshape(1, -1)), axis=0)
    # print(vm_array[-1])

    # get transformer ratings and possibly other properties if first timestep
    if gblvar.it == 0:
        gblvar.trans_rated_s = []
        for i in range(len(gblvar.trans_list)):
            name = gblvar.trans_list[i]
            data = gridlabd.get_object(name)    # USE THIS TO GET ANY OBJECT NEEDED
            # print(data)
            trans_config_name = data['configuration']
            data = gridlabd.get_object(trans_config_name)
            gblvar.trans_rated_s.append(float(data['power_rating'].split(' ')[0]))
        # print(gblvar.trans_rated_s)

    # get transformer power from previous timestep
    gblvar.trans_power = []
    for i in range(len(gblvar.trans_list)):
        name = gblvar.trans_list[i]
        data = gridlabd.get_object(name)
        trans_power_str = data['power_in']
        # print(trans_power_str)
        pmag, pdeg = get_trans_power(trans_power_str)
        gblvar.trans_power.append(pmag / 1000)  # in units kVA

    ####################### SIMULATE ##################################

    # propagage transformer state
    sim.sim_transformer()

    ### add in battery simulation here

    ########################### OPTIMIZE #####################################################

    # xxx = opt.opt_dummy()   # this is done in the controller Optimize abstraction

    ################################# CALCULATE POWER INJECTIONS FOR GRIDLABD ##########################################

    # calculate base_power and pf quantities to set for this timestep
    name_list_base_power = list(gblvar.p_df.columns)
    set_power_vec = np.zeros((len(name_list_base_power),), dtype=complex)
    for i in range(len(name_list_base_power)):
        set_power_vec[i] = gblvar.p_df[name_list_base_power[i]][gblvar.it] + gblvar.q_df[name_list_base_power[i]][
            gblvar.it] * 1j

    # get loads from EV charging station
    num_steps = 1
    # print("Global time is: ", gblvar.it)
    if gblvar.it % EV_charging_sim.resolution == 0:
        """only step when controller time matches pf..based on resolution.
        This ensures allows for varied resolution for ev-charging vs pf solver"""
        charging_net_loads_per_loc = EV_charging_sim.step(num_steps)

    ################################## SEND TO GRIDLABD ################################################

    # set base_power properties for this timestep, i.e. existing loads, home base loads, buildings, etc.
    charging_nodes = EV_charging_sim.get_charging_sites()
    if central_storage:
        central_storage_nodes = EV_charging_sim.get_storage_sites()
    for i in range(len(name_list_base_power)):
        name = name_list_base_power[i]
        prop = 'power_12'
        total_node_load = 0
        # if ev node is power node, add ev_charging power to the set value for power vec.
        if name in charging_nodes:
            charger = EV_charging_sim.get_charger_obj_by_loc(name)
            # charger_load = charger.get_current_load()
            total_node_load += charger.get_current_load()
        if central_storage:
            if name in central_storage_nodes:
                storage = EV_charging_sim.get_storage_obj_by_loc(name)
                total_node_load += storage.power  # units in kW (should be negative if there is discharge to the grid/charger)
        gridlabd.set_value(name, prop, str(set_power_vec[i]+total_node_load).replace('(', '').replace(')', ''))

    # set fast charging power properties for this timestep
    total_node_load_Watts = 0   # for dcfc
    for name in charging_nodes:
        charger_load = EV_charging_sim.get_charger_obj_by_loc(name).get_current_load()   # this is in kW
        total_node_load_Watts = charger_load * 1000     # converting to watts
        prop_1 = 'constant_power_A'
        prop_2 = 'constant_power_B'
        prop_3 = 'constant_power_C'
        gridlabd.set_value(name, prop_1, str(total_node_load_Watts / 3))    # balancing dcfc load between 3-phase
        gridlabd.set_value(name, prop_2, str(total_node_load_Watts / 3))
        gridlabd.set_value(name, prop_3, str(total_node_load_Watts / 3))

    # increment timestep
    gblvar.it = gblvar.it + 1
    # if gblvar.it == 1:
    #     os.chdir(save_folder_prefix)
    return True


def on_term(t):
    '''Stuff to do at the very end of the whole simulation, like saving data'''
    # os.chdir(save_folder_prefix)
    # print(os.getcwd())
    global tic
    # EV_charging_sim.load_results_summary(save_folder_prefix)
    np.savetxt(save_folder_prefix+'/volt_mag.txt', gblvar.vm)
    np.savetxt(save_folder_prefix+'/volt_phase.txt', gblvar.vp)
    np.savetxt(save_folder_prefix+'/nom_vmag.txt', gblvar.nom_vmag)
    np.savetxt(save_folder_prefix+'/trans_To.txt', gblvar.trans_To)
    np.savetxt(save_folder_prefix+'/trans_Th.txt', gblvar.trans_Th)
    with open(save_folder_prefix+'/scenario.json', "w") as outfile:
        json.dump(gblvar.scenario, outfile)
    # pd.DataFrame(data=gblvar.trans_Th, columns=gblvar.trans_list).to_csv(save_folder_prefix+'/trans_Th.csv')
    # pd.DataFrame(data=gblvar.nom_vmag, columns=gblvar.voltage_obj).to_csv(save_folder_prefix+'/nom_vmag.csv')
    toc = time.time()
    print("Total run time: ", (toc - tic) / 60, "minutes")
    gridlabd.cancel()
    # gridlabd.pause()
    return True

def find(criteria):
    '''Finding objects in gridlabd that satisfy certain criteria'''

    finder = criteria.split("=")
    if len(finder) < 2:
        raise Exception("find(criteria='key=value'): criteria syntax error")
    objects = gridlabd.get("objects")
    result = []
    for name in objects:
        item = gridlabd.get_object(name)
        if finder[0] in item and item[finder[0]] == finder[1]:
            if "name" in item.keys():
                result.append(item["name"])
            else:
                result.append("%s:%s" % (item["class"], item["id"]))
    return result


def get_voltage():
    '''Get voltage string from GridLAB-D and process it into float'''

    vm_array = np.zeros((len(gblvar.voltage_obj),))
    vp_array = np.zeros((len(gblvar.voltage_prop),))
    for i in range(len(gblvar.voltage_obj)):
        name = gblvar.voltage_obj[i]
        prop = gblvar.voltage_prop[i]
        data = gridlabd.get_object(name)
        if 'e-' in data[prop]:
            if 'd' in data[prop]:
                data[prop] = data[prop].replace('e-', '(')
                # print(data[prop])
                vl = data[prop].rstrip('d V').replace('+', ',+').replace('-', ',-').split(',')
                if '(' in vl[1]:
                    vl[1] = vl[1].replace('(', 'e-')
                else:
                    pass
                if '(' in vl[2]:
                    vl[2] = vl[2].replace('(', 'e-')
                else:
                    pass
                vm_array[i] = float(vl[1])
                vp_array[i] = float(vl[2])
            else:
                x
        elif 'd' in data[prop]:
            vl = data[prop].rstrip('d V').replace('+', ',+').replace('-', ',-').split(',')
            vm_array[i] = float(vl[1])
            vp_array[i] = float(vl[2])

        else:
            vl = data[prop].rstrip('j V').replace('+', ',+').replace('-', ',-').split(',')

            vm_array[i] = (float(vl[1]) ** 2 + float(vl[2]) ** 2) ** 0.5
            vp_array[i] = np.rad2deg(np.angle(float(vl[1]) + float(vl[2]) * 1j))
    return vm_array, vp_array


def get_trans_power(trans_power_str):
    '''Get power at transformer as a string and process it into a float'''

    trans_power_str = trans_power_str.rstrip(' VA')
    if 'e-' in trans_power_str:
        if 'd' in trans_power_str:
            trans_power_str = trans_power_str.replace('e-', '(')
            strtemp = trans_power_str.rstrip('d').replace('+', ',+').replace('-', ',-').split(',')
            if '(' in strtemp[1]:
                strtemp[1] = strtemp[1].replace('(', 'e-')
            else:
                pass
            if '(' in strtemp[2]:
                strtemp[2] = strtemp[2].replace('(', 'e-')
            else:
                pass
            pmag = float(strtemp[1])
            pdeg = float(strtemp[2])
        else:
            x
    elif 'd' in trans_power_str:
        strtemp = trans_power_str.rstrip('d').replace('+', ',+').replace('-', ',-').split(',')
        pmag = float(strtemp[1])
        pdeg = float(strtemp[2])

    else:
        strtemp = trans_power_str.rstrip('j').replace('+', ',+').replace('-', ',-').split(',')

        pmag = (float(strtemp[1]) ** 2 + float(strtemp[2]) ** 2) ** 0.5
        pdeg = np.rad2deg(np.angle(float(strtemp[1]) + float(strtemp[2]) * 1j))

    # print(str(pmag)+'  '+str(pdeg))

    return pmag, pdeg
