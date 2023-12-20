# TODO: remove this file, it is not going to be required anymore.

# import numpy as np
# import pandas as pd
# import pickle
#
# # define global python simulation variables and default initial values
#
# # iteration number
# it = 0  # what does this mean?
#
# # power flow timestep
# pf_dt = 60
#
# # uncontrollable load profiles
# p_df = pd.read_csv('real_power.csv')
# q_df = pd.read_csv('reactive_power.csv')
# print("Done reading base power files")
#
# p_array = np.asarray(p_df)
# q_array = np.asarray(q_df)
#
# # voltage objects and properties
# with open('voltage_obj.txt', 'rb') as fp:
#     voltage_obj = pickle.load(fp)
# with open('voltage_prop.txt', 'rb') as fp:
#     voltage_prop = pickle.load(fp)
# vm = np.zeros((1, len(voltage_obj)))
# vp = np.zeros((1, len(voltage_obj)))
# v_pred = np.zeros((1, len(voltage_obj)))
#
# ####################### RESOURCE PROPERTIES ################################
# # move to other file format, maybe json, generated from feeder population code
#
# # transformer properties
# trans_dt = 10.0  # integration timestep [seconds]?
# trans_Ta = 20.0  # ambient temperature[C]
#
# # TODO: find where all these transformer values were obtained from
# trans_R = 5.0  # ratio of copper loss to iron loss at rated load
# trans_tau_o = 2 * 60 * 60.0  # top oil time constant in seconds
# trans_tau_h = 6 * 60.0  # hotspot time constant in seconds
# trans_n = 0.9  # how did we get these numbers ?
# trans_m = 0.8  # how did we get these numbers ?
# trans_delta_theta_hs_rated = 28.0
# trans_delta_theta_oil_rated = 36.0
#
# trans_To0 = 30.0  # initial oil temperature [C]
# trans_Th0 = 60.0  # initial hot spot temperature [C]    # How is this set?
# trans_int_method = 'euler'  # integration method ['euler' or 'RK4']
#
# # Battery properties
#
# bat_soc0 = 0.5
# cap_E = 13500  # Wh
