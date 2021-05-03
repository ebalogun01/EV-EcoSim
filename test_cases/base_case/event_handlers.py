import os
import numpy as np
import gridlabd
import time
import gblvar
import re
from EVCharging import ChargingSim

EV_charging_sim = ChargingSim(1)    # Initialize Charging Simulation Class

def on_init(t):

    #get object lists from GridLAB-D
    gridlabd.output("timestamp,x")
    gblvar.node_list=find("class=node")
    gblvar.load_list=find("class=load")
    gblvar.tn_list=find("class=triplex_node")
    gblvar.trans_list=find("class=transformer")
    gblvar.transconfig_list=find("class=transformer_configuration")

    return True

def on_precommit(t):
    clock=gridlabd.get_global("clock")
    print('****  '+str(clock)+'  ****')
    EV_charging_sim.setup()     # Sets up the simulation module with Charging sites and batteries
    num_steps = 1
    charging_net_loads_per_loc = EV_charging_sim.step(num_steps)
    #get voltage from GridLAB-D
    vm_array,vp_array=get_voltage()

    if gblvar.it==0:
        gblvar.nom_vmag=vm_array
    if gblvar.it==1:
        gblvar.vm=vm_array.reshape(1,-1)
        gblvar.vp=vp_array.reshape(1,-1)
    elif gblvar.it>1:
        gblvar.vm=np.concatenate((gblvar.vm,vm_array.reshape(1,-1)),axis=0)
        gblvar.vp=np.concatenate((gblvar.vp,vp_array.reshape(1,-1)),axis=0)
    print(vm_array[-1])

            

    #calculate base_power and pf quantities to set for this timestep
    name_list_base_power=list(gblvar.p_df.columns)
    set_power_vec=np.zeros((len(name_list_base_power),),dtype=complex)
    for i in range(len(name_list_base_power)): # add EV simulation net load for each location
        set_power_vec[i]=gblvar.p_df[name_list_base_power[i]][gblvar.it]+gblvar.q_df[name_list_base_power[i]][gblvar.it]*1j +\
                         charging_net_loads_per_loc[i]

    #set base_power properties for this timestep
    for i in range(len(name_list_base_power)):
        name=name_list_base_power[i]
        prop='power_12'
        gridlabd.set_value(name,prop,str(set_power_vec[i]).replace('(','').replace(')',''))

    #increment timestep
    gblvar.it=gblvar.it+1

    return True

def on_term(t):
    np.savetxt('volt_mag.txt',gblvar.vm)
    np.savetxt('volt_phase.txt',gblvar.vp)
    np.savetxt('nom_vmag.txt',gblvar.nom_vmag)



def find(criteria) :
    finder = criteria.split("=")
    if len(finder) < 2 :
        raise Exception("find(criteria='key=value'): criteria syntax error")
    objects = gridlabd.get("objects")
    result = []
    for name in objects :
        item = gridlabd.get_object(name)
        if finder[0] in item and item[finder[0]] == finder[1] :
            if "name" in item.keys() :
                result.append(item["name"])
            else :
                result.append("%s:%s" % (item["class"],item["id"]))
    return result

def get_voltage():
    
    vm_array=np.zeros((len(gblvar.voltage_obj),))
    vp_array=np.zeros((len(gblvar.voltage_prop),))
    for i in range(len(gblvar.voltage_obj)):
        name=gblvar.voltage_obj[i]
        prop=gblvar.voltage_prop[i]
        data = gridlabd.get_object(name)
        if 'e-' in data[prop]:
            if 'd' in data[prop]:
                data[prop]=data[prop].replace('e-','(')
                #print(data[prop])
                vl=data[prop].rstrip('d V').replace('+',',+').replace('-',',-').split(',')
                if '(' in vl[1]:
                    vl[1]=vl[1].replace('(','e-')
                else:
                    pass
                if '(' in vl[2]:
                    vl[2]=vl[2].replace('(','e-')
                else:
                    pass
                vm_array[i]=float(vl[1])
                vp_array[i]=float(vl[2])     
            else:
                x        
        elif 'd' in data[prop]:
             vl=data[prop].rstrip('d V').replace('+',',+').replace('-',',-').split(',')
             vm_array[i]=float(vl[1])
             vp_array[i]=float(vl[2])     
 
        else:
             vl=data[prop].rstrip('j V').replace('+',',+').replace('-',',-').split(',')
               
             vm_array[i]=(float(vl[1])**2+float(vl[2])**2)**0.5     
             vp_array[i]=np.rad2deg(np.angle(float(vl[1])+float(vl[2])*1j))
    return vm_array,vp_array