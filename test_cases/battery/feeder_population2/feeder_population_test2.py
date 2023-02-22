# %%
import glm_mod_functions
import os
import pandas
import datetime
import numpy as np
import ast
import pickle
import random

# read config file
path_prefix = os.getcwd()
os.chdir(path_prefix)  # change directory
path_prefix = path_prefix[0:path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
path_prefix.replace('\\', '/')

f = open('config.txt', 'r')
param_dict = f.read()
f.close()
param_dict = ast.literal_eval(param_dict)

feeder_name = param_dict['feeder_name']
set_sd = param_dict['set_sd']   # what is sd?
mean_scale = param_dict['mean_scale']
base_file_dir = path_prefix+param_dict['base_file_dir']
test_case_dir = path_prefix+param_dict['test_case_dir']
load_data_dir = path_prefix+param_dict['load_data_dir']
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
                glm_dict_base[i]['Control'] = 'MANUAL'

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

# change all load objects to meters (change property names throughout and delete load properties) todo: add rationale
for i in obj_type_base.keys():
    if ('object' in obj_type_base[i].keys()):

        if 'load' in obj_type_base[i]['object']:
            glm_dict_base = glm_mod_functions.replace_load_w_meter(glm_dict_base, glm_dict_base[i]['name'],
                                                                   glm_dict_base[i]['name'].replace('load', 'meter'),
                                                                   obj_type_base)

include_list_base.append('#include "' + feeder_name + '_secondary.glm' + '";')

# delete existing recorders
rec_del_index = []
for i in obj_type_base.keys():
    if ('object' in obj_type_base[i].keys()):
        if 'recorder' in obj_type_base[i]['object']:
            rec_del_index.append(i)
for i in rec_del_index:
    del glm_dict_base[i]
    del obj_type_base[i]

# add dummy player class
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
tape_bool = False
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

# add script on term statement
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
del glm_dict_base[clock_del_index]

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

# find electable meter nodes for imputing fast-charging - take only the 3-phase meter nodes


# write new glm file
print('writing new glm file')
out_dir = test_case_dir
file_name = feeder_name + '_populated.glm'
glm_mod_functions.write_base_glm(glm_dict_base, obj_type_base, globals_list_base, include_list_base, out_dir, file_name,
                                 sync_list_base)

# write voltage objects and property lists
os.chdir(test_case_dir)
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
admd = 3    # todo: add rationale

# % generate glm for homes

# Initialize dictionaries and lists
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

# Triplex line configuration todo: where are these from?
glm_house_dict[key_index] = {'name': 'triplex_line_config',
                             'conductor_1': '''"c1/0 AA triplex"''',
                             'conductor_2': '''"c1/0 AA triplex"''',
                             'conductor_N': '''"c1/0 AA triplex"''',
                             'insulation_thickness': '0.08',
                             'diameter': '0.368'}
obj_type[key_index] = {'object': 'triplex_line_configuration'}
key_index = key_index + 1
# TODO: model two cases for this work. One with 120/240 V and one with
# TODO: find the source for these
#   House Transformer configuration (NOTE: the nominal voltage should depend on the voltage at the node of the spot-load
#   , so a future task will be to automate this. For now, the code doesn't break because the voltages are the same everywhere
glm_house_dict[key_index] = {'name': 'house_transformer',
                             'connect_type': 'SINGLE_PHASE_CENTER_TAPPED',
                             'install_type': 'PADMOUNT',
                             'primary_voltage': str(np.unique(np.array(nominal_voltage))[0]),
                             # update to include possibly multiple transformer configurations
                             'secondary_voltage': '120 V',
                             'power_rating': '20.0',  # units in kVA
                             'resistance': '0.00600',
                             'reactance': '0.00400',
                             'shunt_impedance': '339.610+336.934j'}
obj_type[key_index] = {'object': 'transformer_configuration'}
key_index = key_index + 1

# TODO: add commercial building loads (need to know what voltage feeds into commercial buildings.

####  CONFIGURE LOAD OBJECTS AND TRANSFORMERS FOR DCFC SIMULATION STARTS HERE ####

#   Fast charging station transformer configuration
# need to find only meters with ABC phases (3-phase) for DCFC connection
standard_rating = False
glm_subset_dict_dcfc = {key: subdict for key, subdict in glm_dict_base.items() if
                        'name' in subdict.keys() and 'meter' in subdict['name'] and 'ABC' in subdict['phases']}
num_fast_charging_nodes = param_dict['num_dcfc_nodes']
num_charging_stalls_per_node = 1    # make param later
charging_stall_base_rating = 75  # kW (make param later)
trans_standard_ratings = np.array([3, 6, 9, 15, 30, 37.5, 45, 75, 112.5, 150, 225, 300])  # units in kVA # TODO: get references for these
DCFC_voltage = 480  # volts (480 volts is the most common DCFC transformer Secondary now)
DCFC_trans_power_rating_kW = charging_stall_base_rating * num_charging_stalls_per_node  # kw
load_pf = 0.95  # this can be >= and many EVSE can have pf close to 1. In initial simulation, they will be unity
DCFC_trans_power_rating_kVA = DCFC_trans_power_rating_kW / load_pf
proximal_std_rating = trans_standard_ratings[np.argmin(np.abs(trans_standard_ratings - DCFC_trans_power_rating_kVA))]   # find the closest transformer rating
if standard_rating:
    DCFC_trans_power_rating_kVA = proximal_std_rating
charging_bus_subset_list = random.sample(list(glm_subset_dict_dcfc.values()), num_fast_charging_nodes)

#   TODO: find more accurate properties for the transformer
#   This is the transformer configuration that is inherited for DCFC
glm_house_dict[key_index] = {'name': 'dcfc_transformer',
                             'connect_type': 'WYE_WYE',
                             'install_type': 'PADMOUNT',
                             'primary_voltage': str(np.unique(np.array(nominal_voltage))[0]),
                             'secondary_voltage': str(DCFC_voltage) + ' V',
                             'power_rating': str(DCFC_trans_power_rating_kVA),
                             'resistance': '0.00600',
                             'reactance': '0.00400'
                             }

obj_type[key_index] = {'object': 'transformer_configuration'}
key_index += 1  # this populates a list of all the objects and configs


##########
fast_charging = True
if fast_charging:
    """Not implemented yet, but we want to automate the creation of dedicated transformers if scenario is fast-charging"""
    pass
###############

k = 0

fast_charging_bus_list = []
# getting the 3-phase meters (nodes) that were stored in the charging_bus_subset_list
for meter_dict in charging_bus_subset_list:
    # load object
    glm_house_dict[key_index] = {'name': 'dcfc_load_' + str(k),
                                 'load_class': 'C',
                                 'nominal_voltage': str(DCFC_voltage),
                                 'phases': 'ABCN',
                                 # this phase is currently hard-coded because we know we want to only connect to 3-phase connection
                                 'constant_power_A': '0.0+0.0j',
                                 'constant_power_B': '0.0+0.0j',
                                 'constant_power_C': '0.0+0.0j'}  # the powers get populated in simulation

    # need to check the phase of the load object to determine how to choose the transformer phase

    obj_type[key_index] = {'object': 'load'}
    key_index = key_index + 1

    #   Transformer (DCFC transformer)
    glm_house_dict[key_index] = {'name': 'dcfc_trans_' + str(k),
                                 'phases': meter_dict['phases'],
                                 'from': meter_dict['name'],
                                 'to': 'dcfc_load_' + str(k),
                                 'configuration': 'dcfc_transformer'}
    fast_charging_bus_list.append('dcfc_load_' + str(k))
    obj_type[key_index] = {'object': 'transformer'}
    key_index = key_index + 1
    k = k + 1

os.chdir(test_case_dir)
np.savetxt('dcfc_bus.txt', fast_charging_bus_list, fmt="%s")  # this stores all the nodes in which there is dcfc

# todo: level 2 is 208/240 V so need to reflect that as well

# CREATE TRANSFORMER RECORDERS FOR ONLY NODE WITH CHARGING TO GET POWER FLOWING OUT TO CHECK FOR OVERLOADING


####  CONFIGURE LOAD OBJECTS AND TRANSFORMERS FOR DCFC SIMULATION ENDS HERE ####

######### L2 STARTS HERE ###########
#   This is the transformer configuration that is inherited for L2 208-240V
standard_rating = True
glm_subset_dict_L2 = {key: subdict for key, subdict in glm_dict_base.items() if
                        'name' in subdict.keys() and 'meter' in subdict['name'] and 'ABC' in subdict['phases']}
num_L2_charging_nodes = param_dict['num_L2C_nodes']
num_charging_stalls_per_node = 2   # make param later
charging_stall_base_rating = 11.5  # kW (make param later)
L2_voltage = 240    # Volts (usually 208 - 240V)
L2_trans_power_rating_kW = charging_stall_base_rating * num_charging_stalls_per_node  # kw
load_pf = 0.95  # this can be >= and many EVSE can have pf close to 1. In initial simulation, they will be unity
L2_trans_power_rating_kVA = L2_trans_power_rating_kW / load_pf
proximal_std_rating = trans_standard_ratings[np.argmin(np.abs(trans_standard_ratings - L2_trans_power_rating_kVA))]   # find the closest transformer rating
if standard_rating:
    L2_trans_power_rating_kVA = proximal_std_rating
    print(f'Using standard {L2_trans_power_rating_kVA} kVA rating for L2 chargers')

glm_house_dict[key_index] = {'name': 'L2_transformer',
                             'connect_type': 'SINGLE_PHASE_CENTER_TAPPED',
                             'install_type': 'PADMOUNT',
                             'primary_voltage': str(np.unique(np.array(nominal_voltage))[0]),
                             # update to include possibly multiple transformer configurations
                             'secondary_voltage': '240 V',
                             'power_rating': str(L2_trans_power_rating_kVA),  # units in kVA
                             'resistance': '0.00600',
                             'reactance': '0.00400',
                             'shunt_impedance': '339.610+336.934j'
                             }
obj_type[key_index] = {'object': 'transformer_configuration'}
key_index += 1  # this populates a list of all the objects and configs
######### L2 ENDS (initial trans config ends) HERE ###########

num_transformers_list = []
fraction_commercial_sec_node = 0.1
contains_commercial_load = random.sample(list(range(len(bus_list))), max(int(fraction_commercial_sec_node * len(bus_list)), 1))

# select (sample) triplex node that will have L2 charging in addition to commercial load (e.g. work buildings/hotels)
L2_charging_node_options = []

k = 0
for i in range(len(bus_list)):
    commercial_load = False
    if i in contains_commercial_load:
        commercial_load = True
        num_transformers = int(np.floor(abs(spot_load_list[i]) / (L2_trans_power_rating_kVA * 1000)))
    else:
        num_transformers = int(np.floor(abs(spot_load_list[i]) / (20 * 1000)))  # need to discuss this a bit more todo: 20 was used here because of the tranformer rating is 20
    num_transformers_list.append(num_transformers)
    for j in range(num_transformers):   # number of transformers per bus
        # Triplex node
        if commercial_load:
            num_houses = int(np.floor(param_dict['commercial_building_trans'] * 0.85) * safety_factor / admd)  # admd is the max power. 0.85 is the worst pf
        else:
            num_houses = int(np.floor(20 * 0.85) * safety_factor / admd)  # admd is the max power. 0.85 is the worst pf
        real_power_trans = np.sum(
            data_use_mat[:, np.random.choice(np.arange(data_use_mat.shape[1]), size=(num_houses,))], axis=1)
        pf_trans = np.random.uniform(0.85, 1.0, size=real_power_trans.shape)    # sample power factor between 0.85 and 1.0
        reactive_power_trans = np.multiply(real_power_trans, np.tan(np.arccos(pf_trans)))

        if k == 0:
            real_power_df = pandas.DataFrame({'tn_' + str(k): np.ndarray.flatten(real_power_trans)})
            reactive_power_df = pandas.DataFrame({'tn_' + str(k): reactive_power_trans})
        else:
            real_power_df['tn_' + str(k)] = real_power_trans
            reactive_power_df['tn_' + str(k)] = reactive_power_trans

        if commercial_load:
            glm_house_dict[key_index] = {'name': 'tn_' + str(k),
                                         'nominal_voltage': f'{L2_voltage}.00',
                                         'phases': str(load_phases[i]) + "S",
                                         'power_12': str(spot_load_list[i] / (num_transformers + 3)).replace('(',
                                                                                                             '').replace(
                                             ')',
                                             '')}  # I think the +3 is a bug or an ad-hoc way to make base case run normally
            L2_charging_node_options.append('tn_' + str(k))     # add this into options for L2 charging site for sim
        else:
            glm_house_dict[key_index] = {'name': 'tn_' + str(k),
                                         'nominal_voltage': '120.00',
                                         'phases': str(load_phases[i]) + "S",
                                         'power_12': str(spot_load_list[i] / (num_transformers + 3)).replace('(',
                                                                                                             '').replace(
                                             ')',
                                             '')}  # I think the +3 is a bug or an ad-hoc way to make base case run normally
        obj_type[key_index] = {'object': 'triplex_node'}
        key_index = key_index + 1
        # TODO: reduced the above based on num transformers to ensure that things run normally.

        #   Transformer (triplex transformer)
        if commercial_load:
            glm_house_dict[key_index] = {'name': 'trip_trans_' + str(k),
                                         'phases': str(load_phases[i]) + 'S',
                                         'from': str(bus_list[i]),
                                         'to': 'tn_' + str(k),
                                         'configuration': 'L2_transformer'}
        else:
            glm_house_dict[key_index] = {'name': 'trip_trans_' + str(k),
                                         'phases': str(load_phases[i]) + 'S',
                                         'from': str(bus_list[i]),
                                         'to': 'tn_' + str(k),
                                         'configuration': 'house_transformer'}
        obj_type[key_index] = {'object': 'transformer'}
        key_index = key_index + 1
        k = k + 1

# Now get the desired L2 charging locs
L2_charging_bus_subset_list = random.sample(L2_charging_node_options, num_L2_charging_nodes)
os.chdir(test_case_dir)
np.savetxt('L2charging_bus.txt', L2_charging_bus_subset_list, fmt="%s")

# write out glm file for secondary distribution
out_dir = test_case_dir
file_name = feeder_name + '_secondary.glm'
glm_mod_functions.write_base_glm(glm_house_dict, obj_type, globals_list, include_list, out_dir, file_name, sync_list)

# save load data (EMMANUEL- THESE ARE TYPICALLY UNCONTROLLABLE LOADS)
os.chdir(test_case_dir)
real_power_df.to_csv('real_power.csv', index=False)
reactive_power_df.to_csv('reactive_power.csv', index=False)
