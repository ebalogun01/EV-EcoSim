import os
import sys
import numpy as np
import gridlabd
import time
import gblvar
import re
sys.path.append('/home/ec2-user/EV50_cosimulation/charging_sim')    # change this
# sys.path.append("../../charging_sim")
from EVCharging import ChargingSim
print("*****EV Charging Station Simulation Imported Successfully*****")

EV_charging_sim = ChargingSim(3)  # Initialize Charging Simulation Class with 3 charging site
print("EV_charging_initialized")
# Sets up the simulation module with Charging sites and batteries

global tic, toc
tic = time.time()
def on_init(t):
    # get object lists from GridLAB-D
    # global tic
    # tic = time.time()
    gridlabd.output("timestamp,x")
    gblvar.node_list = find("class=node")
    gblvar.load_list = find("class=load")
    print(gblvar.load_list)
    gblvar.tn_list = find("class=triplex_node")
    gblvar.trans_list = find("class=transformer")
    gblvar.transconfig_list = find("class=transformer_configuration")

    # Configure EV charging simulation
    EV_charging_sim.setup(list(gblvar.tn_list))
    return True


def on_precommit(t):
    clock = gridlabd.get_global("clock")
    print('****  ' + str(clock) + '  ****')
    # get voltage from GridLAB-D
    vm_array, vp_array = get_voltage()

    if gblvar.it == 0:
        gblvar.nom_vmag = vm_array
    if gblvar.it == 1:
        gblvar.vm = vm_array.reshape(1, -1)
        gblvar.vp = vp_array.reshape(1, -1)
    elif gblvar.it > 1:
        gblvar.vm = np.concatenate((gblvar.vm, vm_array.reshape(1, -1)), axis=0)
        gblvar.vp = np.concatenate((gblvar.vp, vp_array.reshape(1, -1)), axis=0)
    print(vm_array[-1])

    # calculate base_power and pf quantities to set for this timestep
    name_list_base_power = list(gblvar.p_df.columns)
    set_power_vec = np.zeros((len(name_list_base_power),), dtype=complex)

    # get loads from EV charging station
    num_steps = 1
    print("Global time is: ", gblvar.it)
    if gblvar.it % EV_charging_sim.resolution == 0:
        """only step when controller time matches pf..based on resolution.
        This ensures varying resolution for ev-charging vs pf solver"""
        charging_net_loads_per_loc = EV_charging_sim.step(num_steps)
        # print("Net load at {} is".format())

    for i in range(len(name_list_base_power)):  # add EV simulation net load for each location
        print("No error")
        set_power_vec[i] = gblvar.p_df[name_list_base_power[i]][gblvar.it] + gblvar.q_df[name_list_base_power[i]][
            gblvar.it] * 1j
        print('NOENE')
    print(gblvar.it, 'Time done')

    # set base_power properties for this timestep
    charging_nodes = EV_charging_sim.get_charging_sites()
    for i in range(len(name_list_base_power)):
        # here
        name = name_list_base_power[i]
        prop = 'power_12' # CAN YOU ADD SOME EXPLANATION FOR THIS PROP. DON'T REALLY UNDERSTAND
        # if ev node is power node, add ev_charging power to the set value for power vec.
        if name in charging_nodes:
            charger = EV_charging_sim.get_charger_obj_by_loc(name)
            charger_load = charger.get_current_load()
            # print('load is', charger_load, "kW")
            gridlabd.set_value(name, prop, str(set_power_vec[i] + charger_load).replace('(', '').replace(')', ''))
        else:
            gridlabd.set_value(name, prop, str(set_power_vec[i] + 0).replace('(', '').replace(')', ''))

    # increment timestep
    gblvar.it = gblvar.it + 1

    return True


def on_term(t):
    global tic
    EV_charging_sim.load_results_summary()
    np.savetxt('volt_mag.txt', gblvar.vm)
    np.savetxt('volt_phase.txt', gblvar.vp)
    np.savetxt('nom_vmag.txt', gblvar.nom_vmag)
    toc = time.time()
    print("Total run time: ", (toc - tic)/60, "minutes")


def find(criteria):
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
