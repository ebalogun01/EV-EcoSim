import numpy as np
import gblvar


def sim_transformer():
    '''
    Propage transformer state from previous timestep to current timestep
    Nonlinear model from Swift (2001)
    '''

    #if first timestep, initialize state of transformer
    if gblvar.it==0:
        gblvar.trans_To=np.ones((1,len(gblvar.trans_list)))*float(gblvar.trans_To0)
        gblvar.trans_Th=np.ones((1,len(gblvar.trans_list)))*float(gblvar.trans_Th0)

    # Propagate transformer thermal state from previous timestep to current timestep
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

    return -1


