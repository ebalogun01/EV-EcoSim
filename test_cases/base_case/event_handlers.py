import os
import sys

print("os import")
import numpy as np
import gridlabd

print("gridlab imported")
import time
import gblvar

print("glbvar importedgbl")
import re

print("initial imports ok")
sys.path.append('/home/ec2-user/EV50_cosimulation/charging_sim')
# sys.path.append("../../charging_sim")
print('ok append')
from EVCharging import ChargingSim

print("*****EV Charging Station Simulation Imported Successfully*****")

EV_charging_sim = ChargingSim(30)  # Initialize Charging Simulation Class with one charging site
  # Sets up the simulation module with Charging sites and batteries


def on_init(t):
    # get object lists from GridLAB-D
    gridlabd.output("timestamp,x")
    gblvar.node_list = find("class=node")
    # EV_charging_sim.nodes = gblvar.node_list  # add node names into ev_simulation
    # print(gblvar.node_list)
    gblvar.load_list = find("class=load")
    print(gblvar.load_list)
    gblvar.tn_list = find("class=triplex_node")
    gblvar.trans_list = find("class=transformer")
    gblvar.transconfig_list = find("class=transformer_configuration")

    # Configure EV charging simulation
    EV_charging_sim.setup(list(gblvar.p_df.columns))
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
    charging_net_loads_per_loc = EV_charging_sim.step(num_steps)

    for i in range(len(name_list_base_power)):  # add EV simulation net load for each location
        set_power_vec[i] = gblvar.p_df[name_list_base_power[i]][gblvar.it] + gblvar.q_df[name_list_base_power[i]][
            gblvar.it] * 1j
    print(gblvar.it, '1 done')

    # set base_power properties for this timestep
    charging_nodes = EV_charging_sim.get_charging_sites()
    print('charging nodes is', charging_nodes)
    for i in range(len(name_list_base_power)):
        # here
        name = name_list_base_power[i]
        print("NAME", name)
        prop = 'power_12'
        # if ev node is power node, add ev_charging power to the set value for power vec.
        if name in charging_nodes:
            charger = EV_charging_sim.get_charger_obj_by_loc(name)
            charger_load = charger.get_current_load()
            print('load is', charger_load)
            gridlabd.set_value(name, prop, str(set_power_vec[i] + charger_load).replace('(', '').replace(')', ''))
        else:
            gridlabd.set_value(name, prop, str(set_power_vec[i] + 0).replace('(', '').replace(')', ''))

    # increment timestep
    gblvar.it = gblvar.it + 1

    return True


def on_term(t):
    np.savetxt('volt_mag.txt', gblvar.vm)
    np.savetxt('volt_phase.txt', gblvar.vp)
    np.savetxt('nom_vmag.txt', gblvar.nom_vmag)


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
