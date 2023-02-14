# %%

import glm_mod_functions
import os
import pandas
import datetime
import numpy as np
import ast
import pickle

# read config file
path_prefix = os.getcwd()
path_prefix = path_prefix[0:path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
path_prefix.replace('\\', '/')
os.chdir(path_prefix + '/feeder_population')  # change directory
f = open('config.txt', 'r')
param_dict = f.read()
f.close()
param_dict = ast.literal_eval(param_dict)

feeder_name = param_dict['feeder_name']
set_sd = param_dict['set_sd']
mean_scale = param_dict['mean_scale']
base_file_dir = path_prefix + param_dict['base_file_dir']
test_case_dir = path_prefix + param_dict['test_case_dir']
load_data_dir = path_prefix + param_dict['load_data_dir']
box_pts = param_dict['box_pts']
starttime_str = param_dict['starttime']
endtime_str = param_dict['endtime']
python_module = param_dict['python_module']
safety_factor = param_dict['safety_factor']

base_glm_file = feeder_name + '.glm'
print('Loading original glm')
glm_dict_base, obj_type_base, globals_list_base, include_list_base, sync_list_base = glm_mod_functions.load_base_glm(
    base_file_dir, base_glm_file)

print('Modifying properties')
spot_load_list = []
bus_list = []
bus_list_voltage = []
prop_voltage = []
nominal_voltage = []
load_phases = []

for i in obj_type_base.keys():
    if ('object' in obj_type_base[i].keys()):

        # modify load objects
        if 'load' in obj_type_base[i]['object']:
            if 'constant_power_A' in glm_dict_base[i].keys():
                spot_load_list.append(complex(glm_dict_base[i]['constant_power_A']))
                bus_list.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"').replace('load', 'meter'))
                nominal_voltage.append(glm_dict_base[i]['nominal_voltage'])
                load_phases.append('A')

            if 'A' in glm_dict_base[i]['phases']:
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"').replace('load', 'meter'))
                prop_voltage.append('voltage_A')

            if 'constant_power_B' in glm_dict_base[i].keys():
                spot_load_list.append(complex(glm_dict_base[i]['constant_power_B']))
                bus_list.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"').replace('load', 'meter'))
                nominal_voltage.append(glm_dict_base[i]['nominal_voltage'])
                load_phases.append('B')

            if 'B' in glm_dict_base[i]['phases']:
                prop_voltage.append('voltage_B')
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"').replace('load', 'meter'))

            if 'constant_power_C' in glm_dict_base[i].keys():
                spot_load_list.append(complex(glm_dict_base[i]['constant_power_C']))
                bus_list.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"').replace('load', 'meter'))
                nominal_voltage.append(glm_dict_base[i]['nominal_voltage'])
                load_phases.append('C')

            if 'C' in glm_dict_base[i]['phases']:
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"').replace('load', 'meter'))
                prop_voltage.append('voltage_C')


        # get rid of regulator control
        elif 'regulator_configuration' in obj_type_base[i]['object']:
            if 'Control' in glm_dict_base[i].keys():
                glm_dict_base[i]['Control'] = 'MANUAL'  # why is control made manual here?

        # get rid of capacitor control
        elif 'capacitor' in obj_type_base[i]['object']:
            if 'control' in glm_dict_base[i].keys():
                glm_dict_base[i]['control'] = 'MANUAL'
            if 'A' in glm_dict_base[i]['phases_connected']:
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"'))
                prop_voltage.append('voltage_A')
            if 'B' in glm_dict_base[i]['phases_connected']:
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"'))
                prop_voltage.append('voltage_B')
            if 'C' in glm_dict_base[i]['phases_connected']:
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"'))
                prop_voltage.append('voltage_C')

        elif 'node' in obj_type_base[i]['object']:
            if 'A' in glm_dict_base[i]['phases']:
                # print(glm_dict_base[i]['phases'])
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"'))
                prop_voltage.append('voltage_A')
            if 'B' in glm_dict_base[i]['phases']:
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"'))
                prop_voltage.append('voltage_B')
            if 'C' in glm_dict_base[i]['phases']:
                bus_list_voltage.append(glm_dict_base[i]['name'].rstrip('"').lstrip('"'))
                prop_voltage.append('voltage_C')

# change all load objects to meters (change property names throughout and delete load properties)
for i in obj_type_base.keys():
    if ('object' in obj_type_base[i].keys()):

        if 'load' in obj_type_base[i]['object']:
            glm_dict_base = glm_mod_functions.replace_load_w_meter(glm_dict_base, glm_dict_base[i]['name'],
                                                                   glm_dict_base[i]['name'].replace('load', 'meter'),
                                                                   obj_type_base)

include_list_base.append(
    '#include "' + feeder_name + '_secondary.glm' + '";')  # adds the secondary distribution in simulation

# delete existing recorders
rec_del_index = []
for i in obj_type_base.keys():
    if ('object' in obj_type_base[i].keys()):
        if 'recorder' in obj_type_base[i]['object']:
            rec_del_index.append(i)
for i in rec_del_index:
    del glm_dict_base[i]
    del obj_type_base[i]

# add dummy player class (this allows running multiple time-steps)
key_index = max(glm_dict_base.keys()) + 1
obj_type_base[key_index] = {'class': 'dummy'}
glm_dict_base[key_index] = {'double': 'value'}

key_index = max(glm_dict_base.keys()) + 1
obj_type_base[key_index] = {'class': 'player'}
glm_dict_base[key_index] = {'double': 'value'}

key_index = max(glm_dict_base.keys()) + 1
obj_type_base[key_index] = {'object': 'player'}
glm_dict_base[key_index] = {'name': 'dummy_player',
                            'file': '"dummy.player"'}

key_index = max(glm_dict_base.keys()) + 1
obj_type_base[key_index] = {'object': 'dummy'}
glm_dict_base[key_index] = {'name': 'dummy_obj',
                            'value': 'dummy_player.value'}

# add tape module if not already there
tape_bool = False  # double-check what the tape module does
for i in obj_type_base.keys():
    if ('module' in obj_type_base[i].keys()):
        if 'tape' in obj_type_base[i]['module']:
            tape_bool = True
if tape_bool == False:
    key_index = max(glm_dict_base.keys()) + 1
    obj_type_base[key_index] = {'module': 'tape'}
    glm_dict_base[key_index] = {}

# add python module
key_index = max(glm_dict_base.keys()) + 1
obj_type_base[key_index] = {'module': python_module}
glm_dict_base[key_index] = {}

# add script on term statement (CAN MODIFY THIS)
sync_list_base.append('script on_term "python3 voltdump2.py";')

# add voltdump
key_index = max(glm_dict_base.keys()) + 1
obj_type_base[key_index] = {'object': 'voltdump'}
glm_dict_base[key_index] = {'name': '"voltdump"',
                            'filemode': '"a"',
                            'filename': '"volt_dump.csv"',
                            'interval': '60',
                            'version': '1'}

# check if minimum timestep is already set
if ('#set minimum_timestep=60' in globals_list_base) == False:
    globals_list_base.append('#set minimum_timestep=60')

# delete clock object
for i in obj_type_base.keys():
    if ('clock' in obj_type_base[i].keys()):
        clock_del_index = i
del glm_dict_base[clock_del_index]  # why isn't this in the for-loop?

# add new clock object
glm_dict_base[clock_del_index] = {'starttime': starttime_str,
                                  'stoptime': endtime_str}

# remove powerflow object
for i in obj_type_base.keys():
    if ('module' in obj_type_base[i].keys()):
        if 'powerflow' in obj_type_base[i]['module']:
            pf_del_index = i
del glm_dict_base[pf_del_index]

# add new powerflow object that outputs NR solver information
glm_dict_base[pf_del_index] = {'solver_method': 'NR',
                               'line_capacitance': 'true',
                               'convergence_error_handling': 'IGNORE',
                               'solver_profile_enable': 'true',
                               'solver_profile_filename': '"solver_nr_out.csv"'}

# write new glm file
print('writing new glm file')
out_dir = test_case_dir
file_name = feeder_name + '_populated.glm'
glm_mod_functions.write_base_glm(glm_dict_base, obj_type_base, globals_list_base, include_list_base, out_dir, file_name,
                                 sync_list_base)

# write voltage objects and property lists
with open('voltage_obj.txt', 'wb') as fp:
    pickle.dump(bus_list_voltage, fp)
with open('voltage_prop.txt', 'wb') as fp:
    pickle.dump(prop_voltage, fp)

# % load residential load data

os.chdir(load_data_dir)
data_use = pandas.read_csv('data_2015_use.csv')

year = 2018

timestamp_list = [[] for k in range(len(data_use.month))]
for i in range(len(timestamp_list)):
    timestamp_list[i] = datetime.datetime(year, data_use.month[i],
                                          data_use.day[i], data_use.hour[i],
                                          data_use.minute[i])
data_use['timestamp'] = [datetime.datetime.strftime(k, "%m-%d-%Y %H:%M:%S") for k in timestamp_list]
data_use = data_use.set_index(pandas.DatetimeIndex(data_use.timestamp))

start_time = datetime.datetime(int(starttime_str[1:5]), int(starttime_str[6:8]), int(starttime_str[9:11]),
                               int(starttime_str[12:14]), int(starttime_str[15:17]))
end_time = datetime.datetime(int(endtime_str[1:5]), int(endtime_str[6:8]), int(endtime_str[9:11]),
                             int(endtime_str[12:14]), int(endtime_str[15:17])) + datetime.timedelta(minutes=1)

data_use_filt = data_use[data_use.index >= start_time]
data_use_filt = data_use_filt[data_use_filt.index < end_time]

data_use_mat = np.asarray(data_use_filt[data_use.columns[6:-1]]) * 1000
agg_power = np.mean(data_use_mat, axis=1)
admd = np.max(agg_power)
admd = 3

# % generate glm for homes

# Initiatize dictionaries and lists
glm_house_dict = {}
obj_type = {}
globals_list = []
include_list = []
sync_list = []

key_index = 0

glm_house_dict[key_index] = {}
obj_type[key_index] = {'module': 'tape'}
key_index = key_index + 1

# Triplex line conductor
glm_house_dict[key_index] = {'name': '''"c1/0 AA triplex"''',
                             'resistance': '0.97',
                             'geometric_mean_radius': '0.0111'}

obj_type[key_index] = {'object': 'triplex_line_conductor'}
key_index = key_index + 1

# Triplex line configuration
glm_house_dict[key_index] = {'name': 'triplex_line_config',
                             'conductor_1': '''"c1/0 AA triplex"''',
                             'conductor_2': '''"c1/0 AA triplex"''',
                             'conductor_N': '''"c1/0 AA triplex"''',
                             'insulation_thickness': '0.08',
                             'diameter': '0.368'}
obj_type[key_index] = {'object': 'triplex_line_configuration'}
key_index = key_index + 1

# Transformer configuration
glm_house_dict[key_index] = {'name': 'house_transformer',
                             'connect_type': 'SINGLE_PHASE_CENTER_TAPPED',
                             'install_type': 'PADMOUNT',
                             'primary_voltage': str(np.unique(np.array(nominal_voltage))[0]),
                             # update to include possibly multiple transformer configurations
                             'secondary_voltage': '120 V',
                             'power_rating': '20.0',
                             'resistance': '0.00600',
                             'reactance': '0.00400',
                             'shunt_impedance': '339.610+336.934j'}
obj_type[key_index] = {'object': 'transformer_configuration'}
key_index = key_index + 1

num_transformers_list = []
k = 0
for i in range(len(bus_list)):
    num_transformers = int(np.floor(abs(spot_load_list[i]) / (
                20 * 1000)))  # is this 20kVA based on the defined rating? change this from hard-coded?
    num_transformers_list.append(num_transformers)
    for j in range(num_transformers):
        # Triplex node
        num_houses = int(np.floor(20 * 0.85) * safety_factor / admd)
        real_power_trans = np.sum(
            data_use_mat[:, np.random.choice(np.arange(data_use_mat.shape[1]), size=(num_houses,))], axis=1)
        pf_trans = np.random.uniform(0.85, 1.0,
                                     size=real_power_trans.shape)  # randomly sample power factor for the homes load for transformer, make the lower limit generic as well
        reactive_power_trans = np.multiply(real_power_trans, np.tan(np.arccos(pf_trans)))

        if k == 0:
            real_power_df = pandas.DataFrame({'tn_' + str(k): np.ndarray.flatten(real_power_trans)})
            reactive_power_df = pandas.DataFrame({'tn_' + str(k): reactive_power_trans})
        else:
            real_power_df['tn_' + str(k)] = real_power_trans
            reactive_power_df['tn_' + str(k)] = reactive_power_trans

        glm_house_dict[key_index] = {'name': 'tn_' + str(k),
                                     'nominal_voltage': '120.00',
                                     'phases': str(load_phases[i]) + "S",
                                     'power_12': str(spot_load_list[i] / (num_transformers + 3)).replace('(',
                                                                                                         '').replace(
                                         ')', '')}
        obj_type[key_index] = {'object': 'triplex_node'}
        key_index = key_index + 1

        # Transformer
        glm_house_dict[key_index] = {'name': 'trip_trans_' + str(k),
                                     'phases': str(load_phases[i]) + 'S',
                                     'from': str(bus_list[i]),
                                     'to': 'tn_' + str(k),
                                     'configuration': 'house_transformer'}
        obj_type[key_index] = {'object': 'transformer'}
        key_index = key_index + 1
        k = k + 1

# write out glm file for secondary distribution
out_dir = test_case_dir
file_name = feeder_name + '_secondary.glm'
glm_mod_functions.write_base_glm(glm_house_dict, obj_type, globals_list, include_list, out_dir, file_name, sync_list)

# save load data
os.chdir(test_case_dir)
real_power_df.to_csv('real_power.csv', index=False)
reactive_power_df.to_csv('reactive_power.csv', index=False)
