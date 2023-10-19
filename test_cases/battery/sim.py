"""This file simulates the transformer thermal state by propagating it forward at each time step """

import numpy as np
import gblvar
import sys
sys.path.append('../..')


def sim_transformer(temperature_data=None):
    """
    Propagate transformer state from previous timestep to current timestep (currently it is a minute resolution data
    simulated in 10 seconds increment).
    Nonlinear model from Swift et. Al (2001).
    Inputs: temperature_data - ambient temperature data for a given region.
    Returns: -1, just arbitrarily.
    """
    # todo: make this a class

    #   if first timestep, initialize state of transformer
    if gblvar.it == 0:
        gblvar.trans_To = np.ones((1, len(gblvar.trans_list))) * float(gblvar.trans_To0)
        gblvar.trans_Th = np.ones((1, len(gblvar.trans_list))) * float(gblvar.trans_Th0)

    if temperature_data is not None:
        gblvar.trans_Ta = temperature_data[gblvar.it]

    # Propagate transformer thermal state from previous timestep to current timestep
    if gblvar.trans_int_method == 'euler':
        trans_To_new= gblvar.trans_To[gblvar.it - 1, :]
        trans_Th_new= gblvar.trans_Th[gblvar.it - 1, :]

        # loop through transformers
        for i in range(len(gblvar.trans_list)):
            #   integrate across powerflow timestep
            for _ in range(int(gblvar.pf_dt / gblvar.trans_dt)):
                trans_To_new[i]= trans_To_new[i] + gblvar.trans_dt * (((gblvar.trans_R * (
                        gblvar.trans_power[i] / gblvar.trans_rated_s[i]) ** 2 + 1) / (gblvar.trans_R + 1))
                                                                      * ((gblvar.trans_delta_theta_oil_rated ** (1 / gblvar.trans_n)) / gblvar.trans_tau_o)
                                                                      - (1 / gblvar.trans_tau_o) * (max(trans_To_new[i] - gblvar.trans_Ta, 0)) ** (1 / gblvar.trans_n))  # top oil temperature

                trans_Th_new[i]= trans_Th_new[i] + gblvar.trans_dt * (((
                                                                               gblvar.trans_power[i] / gblvar.trans_rated_s[i]) ** 2) *
                                                                      ((gblvar.trans_delta_theta_hs_rated ** (1 / gblvar.trans_m)) / (
                                                                          gblvar.trans_tau_h))
                                                                      - (1 / gblvar.trans_tau_h) * (max(trans_Th_new[i] - trans_To_new[i], 0)) ** (1 / gblvar.trans_m))  # hot-spot temperature
        #   append to full data array of temperatures
        gblvar.trans_To = np.concatenate((gblvar.trans_To, trans_To_new.reshape(1, -1)), axis=0)   # maybe use list ?
        gblvar.trans_Th = np.concatenate((gblvar.trans_Th, trans_Th_new.reshape(1, -1)), axis=0)

    return -1


