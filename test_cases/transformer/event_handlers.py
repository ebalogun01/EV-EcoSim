import os
import numpy as np
import gridlabd
import time
import gblvar
import re

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



    #get transformer ratings and possibly other properties if first timestep
    if gblvar.it==0:
        gblvar.trans_rated_s=[]
        for i in range(len(gblvar.trans_list)):
            name=gblvar.trans_list[i]
            data = gridlabd.get_object(name)
            print(data)
            trans_config_name=data['configuration']
            data = gridlabd.get_object(trans_config_name)
            gblvar.trans_rated_s.append(float(data['power_rating'].split(' ')[0]))
        print(gblvar.trans_rated_s)

    #get transformer power
    gblvar.trans_power=[]
    for i in range(len(gblvar.trans_list)):
        name=gblvar.trans_list[i]
        data = gridlabd.get_object(name)
        trans_power_str=data['power_in']
        #print(trans_power_str)
        pmag,pdeg=get_trans_power(trans_power_str)
        gblvar.trans_power.append(pmag/1000) # in units kVA


    #propagate transformer thermal state 
    #if first timestep, initialize state
    if gblvar.it==0:
        gblvar.trans_To=np.ones((1,len(gblvar.trans_list)))*float(gblvar.trans_To0)
        gblvar.trans_Th=np.ones((1,len(gblvar.trans_list)))*float(gblvar.trans_Th0)
    else:
        if gblvar.trans_int_method=='euler':

            trans_To_new=gblvar.trans_To[gblvar.it-1,:]
            trans_Th_new=gblvar.trans_Th[gblvar.it-1,:]

            # loop through transformers

            for i in range(len(gblvar.trans_list)):
                #integrate across powerflow timestep
                for j in range(int(gblvar.pf_dt/gblvar.trans_dt)):
                    trans_To_new[i]=trans_To_new[i]+gblvar.trans_dt*(((gblvar.trans_R*(gblvar.trans_power[i]/gblvar.trans_rated_s[i])**2+1)/(gblvar.trans_R+1))*((gblvar.trans_delta_theta_oil_rated**(1/gblvar.trans_n))/gblvar.trans_tau_o)-(1/gblvar.trans_tau_o)*(max(trans_To_new[i]-gblvar.trans_Ta,0))**(1/gblvar.trans_n))
                    trans_Th_new[i]=trans_Th_new[i]+gblvar.trans_dt*(((gblvar.trans_power[i]/gblvar.trans_rated_s[i])**2)*((gblvar.trans_delta_theta_hs_rated**(1/gblvar.trans_m))/(gblvar.trans_tau_h))-(1/gblvar.trans_tau_h)*(max(trans_Th_new[i]-trans_To_new[i],0))**(1/gblvar.trans_m))
            #append to full data array of temperatures
            gblvar.trans_To=np.concatenate((gblvar.trans_To,trans_To_new.reshape(1,-1)),axis=0)
            gblvar.trans_Th=np.concatenate((gblvar.trans_Th,trans_Th_new.reshape(1,-1)),axis=0)


    #calculate base_power and pf quantities to set for this timestep
    name_list_base_power=list(gblvar.p_df.columns)
    set_power_vec=np.zeros((len(name_list_base_power),),dtype=complex)
    for i in range(len(name_list_base_power)):
        set_power_vec[i]=gblvar.p_df[name_list_base_power[i]][gblvar.it]+gblvar.q_df[name_list_base_power[i]][gblvar.it]*1j

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
    np.savetxt('trans_To.txt',gblvar.trans_To)
    np.savetxt('trans_Th.txt',gblvar.trans_Th)



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

def get_trans_power(trans_power_str):

    trans_power_str=trans_power_str.rstrip(' VA')
    if 'e-' in trans_power_str:
        if 'd' in trans_power_str:
            trans_power_str=trans_power_str.replace('e-','(')
            strtemp=trans_power_str.rstrip('d').replace('+',',+').replace('-',',-').split(',')
            if '(' in strtemp[1]:
                strtemp[1]=strtemp[1].replace('(','e-')
            else:
                pass
            if '(' in strtemp[2]:
                strtemp[2]=strtemp[2].replace('(','e-')
            else:
                pass
            pmag=float(strtemp[1])
            pdeg=float(strtemp[2])     
        else:
            x        
    elif 'd' in trans_power_str:
         strtemp=trans_power_str.rstrip('d').replace('+',',+').replace('-',',-').split(',')
         pmag=float(strtemp[1])
         pdeg=float(strtemp[2])     

    else:
         strtemp=trans_power_str.rstrip('j').replace('+',',+').replace('-',',-').split(',')
              
         pmag=(float(strtemp[1])**2+float(strtemp[2])**2)**0.5     
         pdeg=np.rad2deg(np.angle(float(strtemp[1])+float(strtemp[2])*1j))

    #print(str(pmag)+'  '+str(pdeg))

    return pmag,pdeg
