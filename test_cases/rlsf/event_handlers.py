import numpy as np
import gridlabd
import gblvar


def on_init(t):

    #get object lists from GridLAB-D
    gridlabd.output("timestamp,x")
    gblvar.node_list=find("class=node_name")
    gblvar.load_list=find("class=load")
    gblvar.tn_list=find("class=triplex_node")
    gblvar.trans_list=find("class=transformer")
    gblvar.transconfig_list=find("class=transformer_configuration")

    return True

def on_precommit(t):
    clock=gridlabd.get_global("clock")
    print('****  '+str(clock)+'  ****')

    # make prediction for solution of previous timestep
    if gblvar.it>0:
        y_hat=np.matmul(gblvar.x_array_aug[gblvar.it - 1, :].reshape(1, -1), gblvar.w).reshape(1, -1)
        if gblvar.it==1:
            gblvar.v_pred=y_hat
        else:
            gblvar.v_pred=np.concatenate((gblvar.v_pred, y_hat), axis=0)


    #get voltage from GridLAB-D
    vm_array,vp_array=get_voltage()

    if gblvar.it==0:
        gblvar.nom_vmag=vm_array
    if gblvar.it==1:
        gblvar.vm=vm_array.reshape(1, -1)
        gblvar.vp=vp_array.reshape(1, -1)
    elif gblvar.it>1:
        gblvar.vm=np.concatenate((gblvar.vm, vm_array.reshape(1, -1)), axis=0)
        gblvar.vp=np.concatenate((gblvar.vp, vp_array.reshape(1, -1)), axis=0)
    print(vm_array[-1])

    #calculate error in prediction, update RLSF parameters
    if gblvar.it>0:
        e=vm_array.reshape(1,-1)-y_hat
        rmse_vmag_temp= (np.mean((e[0,0:int(e.shape[1])] / gblvar.nom_vmag.reshape(1, -1)) ** 2)) ** 0.5
        gblvar.rmse_vmag.append(rmse_vmag_temp)
        print(rmse_vmag_temp)
        gblvar.Q= (1 / gblvar.lam) * (gblvar.Q - (np.outer(np.matmul(gblvar.Q,
                                                                     gblvar.x_array_aug[gblvar.it - 1, :].reshape(-1, 1)), np.matmul(
            gblvar.Q, gblvar.x_array_aug[gblvar.it - 1, :].reshape(-1, 1)))) / (
                                              gblvar.lam + np.dot(gblvar.x_array_aug[gblvar.it - 1, :], np.matmul(
                                              gblvar.Q, gblvar.x_array_aug[gblvar.it - 1, :].reshape(-1, 1)))))
        gblvar.w= gblvar.w + np.matmul(np.matmul(gblvar.Q, gblvar.x_array_aug[gblvar.it - 1, :].reshape(-1, 1)), e)


    #calculate base_power and pf quantities to set for this timestep
    name_list_base_power=list(gblvar.p_df.columns)
    set_power_vec=np.zeros((len(name_list_base_power),),dtype=complex)
    for i in range(len(name_list_base_power)):
        set_power_vec[i]= gblvar.p_df[name_list_base_power[i]][gblvar.it] + gblvar.q_df[name_list_base_power[i]][
            gblvar.it] * 1j

    #set base_power properties for this timestep
    for i in range(len(name_list_base_power)):
        name=name_list_base_power[i]
        prop='power_12'
        gridlabd.set_value(name,prop,str(set_power_vec[i]).replace('(','').replace(')',''))

    #increment timestep
    gblvar.it= gblvar.it + 1

    return True

def on_term(t):
    np.savetxt('volt_mag.txt', gblvar.vm)
    np.savetxt('volt_phase.txt', gblvar.vp)
    np.savetxt('nom_vmag.txt', gblvar.nom_vmag)
    np.savetxt('v_pred.txt', gblvar.v_pred)
    np.savetxt('rmse_vmag.txt', np.array(gblvar.rmse_vmag))



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
        name= gblvar.voltage_obj[i]
        prop= gblvar.voltage_prop[i]
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