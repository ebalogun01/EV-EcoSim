"""This file defines some global variables used by the event_handlers.py file. This allows ease of variable sharing
between the relevant file
Inputs: real_power.csv
        reactive_power.csv"""

import numpy as np
import pandas as pd
import pickle

# define global python simulation variables and default initial values

#   flag for scenarios
charging_sim_path_append = False

#   simulation file path
sim_file_path = ''

# temperature data
# month = 1   # todo: think of how to load in the month here
# s = pd.read_csv('SolarData/SolarData_SanJose_2018-30min.csv', index_col='date')
# cols = ['Month', 'Day', 'Hour', 'Temperature']
# temps_data = pd.read_csv(temperature_data_path)[cols]
# temps_data = temps_data[temps_data["Month"] == month]['Temperature']   # for one month


# iteration number
it = 0

# power flow timestep
pf_dt = 60

# uncontrollable load profiles (units W and Var)
p_df = pd.read_csv('real_power.csv')
q_df = pd.read_csv('reactive_power.csv')
p_array = np.asarray(p_df)
q_array = np.asarray(q_df)

# voltage objects and properties
with open('voltage_obj.txt', 'rb') as fp:
    voltage_obj = pickle.load(fp)
with open('voltage_prop.txt', 'rb') as fp:
    voltage_prop = pickle.load(fp)

vm = np.zeros((1, len(voltage_obj)))
vp = np.zeros((1, len(voltage_obj)))
v_pred = np.zeros((1, len(voltage_obj)))
trans_Th = None
trans_list = None
trans_To = None
trans_loading_percent = 0
nom_vmag = None


####################### RESOURCE PROPERTIES ################################
# move to other file format, maybe json, generated from feeder population code

# transformer properties
trans_dt = 10.0  # integration timestep [seconds]?
trans_Ta = 20.0  # ambient temperature[C] {SLIGHTLY HIGHER THAN JUNE AVERAGE IN 2018}

# TODO: find where all these transformer values were obtained from
# transformer has various cooling modes that determine m and n for transformer
# ONAF: Natural convection flow of oil through windings and radiators. Forced convection flow of air over radiators by fans
# ONAN: Natural convection flow of oil through the windings and radiators. Natural convection flow of air over tank and radiation
trans_R = 5.0   # ratio of copper loss to iron loss at rated load
trans_tau_o = 2 * 60 * 60.0 # top oil time constant in seconds
trans_tau_h = 6 * 60.0  # hotspot time constant in seconds
trans_n = 0.9   # how did we get these numbers ?
trans_m = 0.8   # how did we get these numbers ?
trans_delta_theta_hs_rated = 28.0   # NEED TO DECIDE HOW IMPORTANT THESE WILL BE IN THE PAPER
trans_delta_theta_oil_rated = 36.0  # todo: find good sources for these numbers

trans_To0 = 30.0  # initial oil temperature [C] (assume we start at a point where oil is slightly hotter than ambient)
trans_Th0 = 60.0  # initial hot spot temperature [C]    # How is this set? (should not matter long-term)
trans_int_method = 'euler'  # integration method ['euler' or 'RK4']

