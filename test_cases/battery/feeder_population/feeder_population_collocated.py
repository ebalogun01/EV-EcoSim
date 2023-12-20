"""
**Introduction**\n
This is the feeder population module within the battery test case. This file performs the pre-simulation step for
running EV-EcoSim.\n\n
It takes in a base Gridlab-D Model (GLM) file (for example, `IEEE123.glm`), and modifies that file by including
secondary distribution, home loads, and EV Charging station and transformers.

Once this script is done running, it reads and writes new GLM as <initial_glm_name>_populated.glm and
<initial_glm_name>_secondary.glm, and saves them within the test case folder. These saved files are used to run the
simulation. These files are saved in the 'test_case_dir' field specified in config.txt.

**Input file description** \n
Config `config.txt`: configuration file describing the pre-simulation parameters.
This can be modified directly or with the help of our Graphic User Interface (GUI). The return outputs of this module
are files that are read in to run the EV-EcoSim environment.


**Output file description**\n
`real_power.csv` - Real power; this is residential real load timeseries file per node_name/bus \n
`reactive_power.csv` - Reactive power; this is residential reactive load timeseries file per node_name/bus \n
`dcfc_bus.txt` - DC fast charging bus locations; this is used in co-simulation \n
`L2charging_bus.txt` - L2 charging bus locations; this is used in co-simulation \n
"""

import glm_mod_functions
import os
import pandas
import datetime
import numpy as np
import ast
import pickle
import random


def main():
    """
    Runs the feeder population module. It takes in a base Gridlab-D Model (GLM) file (for example,
    `IEEE123.glm`), and modifies that file by including secondary distribution, home loads, and EV Charging station and
    transformers.

    :return: None.
    """
    path_prefix = str(os.getcwd())
    os.chdir(path_prefix)  # change directory
    # Splitting the path is different for Windows and Linux/MacOS. Need condition to deal with both OS file path styles.
    if '\\' in path_prefix:
        path_prefix = "/".join(path_prefix.split('\\')[:-3])   # Gets absolute path to the root of the project to get the desired files.
    else:
        path_prefix = "/".join(path_prefix.split('/')[:-3])

    f = open('config.txt', 'r')
    param_dict = f.read()
    f.close()
    param_dict = ast.literal_eval(param_dict)

    feeder_name = param_dict['feeder_name']
    set_sd = param_dict['set_sd']  # what is sd?
    mean_scale = param_dict['mean_scale']
    base_file_dir = path_prefix + param_dict['base_file_dir']
    test_case_dir = path_prefix + param_dict['test_case_dir']
    load_data_dir = path_prefix + param_dict['load_data_dir']
    base_load_file = param_dict['base_load_file']
    box_pts = param_dict['box_pts']
    starttime_str = param_dict['starttime']
    endtime_str = param_dict['endtime']
    python_module = param_dict['python_module']
    safety_factor = param_dict['safety_factor']  # this helps to account with loading
    pf_min = param_dict['min_power_factor']  # minimum power factor
    pf_max = param_dict['max_power_factor']  # maximum power factor

    base_glm_file = feeder_name + '.glm'
    print('Loading original glm')
    glm_dict_base, obj_type_base, globals_list_base, include_list_base, sync_list_base = glm_mod_functions.load_base_glm(
        base_file_dir, base_glm_file)

    print('Modifying glm properties...')
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

            elif 'node_name' in obj_type_base[i]['object']:
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
        if 'object' in obj_type_base[i].keys():
            if 'load' in obj_type_base[i]['object']:
                glm_dict_base = glm_mod_functions.replace_load_w_meter(glm_dict_base, glm_dict_base[i]['name'],
                                                                       glm_dict_base[i]['name'].replace('load',
                                                                                                        'meter'),
                                                                       obj_type_base)

    include_list_base.append('#include "' + feeder_name + '_secondary.glm' + '";')

    # delete existing recorders
    rec_del_index = []
    for i in obj_type_base.keys():
        if 'object' in obj_type_base[i].keys():
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
        if 'module' in obj_type_base[i].keys():
            if 'tape' in obj_type_base[i]['module']:
                tape_bool = True
    if not tape_bool:
        key_index = max(glm_dict_base.keys()) + 1
        obj_type_base[key_index] = {'module': 'tape'}
        glm_dict_base[key_index] = {}

    # add python module
    key_index = max(glm_dict_base.keys()) + 1
    obj_type_base[key_index] = {'module': python_module}
    glm_dict_base[key_index] = {}

    # add script on term statement
    # sync_list_base.append('script on_term "python3 voltdump2.py";')

    # add voltdump
    key_index = max(glm_dict_base.keys()) + 1
    obj_type_base[key_index] = {'object': 'voltdump'}
    glm_dict_base[key_index] = {'name': '"voltdump"',
                                'filemode': '"a"',
                                'filename': '"volt_dump.csv"',
                                'interval': '60',
                                'version': '1'}

    # check if minimum timestep is already set
    if '#set minimum_timestep=60' in globals_list_base is False:
        globals_list_base.append('#set minimum_timestep=60')

    # delete clock object
    for i in obj_type_base.keys():
        if 'clock' in obj_type_base[i].keys():
            clock_del_index = i
    del glm_dict_base[clock_del_index]

    # add new clock object
    glm_dict_base[clock_del_index] = {'starttime': starttime_str,
                                      'stoptime': endtime_str}

    # remove powerflow object
    for i in obj_type_base.keys():
        if 'module' in obj_type_base[i].keys():
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
    print('writing new glm file...')
    out_dir = test_case_dir
    file_name = feeder_name + '_populated.glm'
    glm_mod_functions.write_base_glm(glm_dict_base, obj_type_base, globals_list_base, include_list_base, out_dir,
                                     file_name,
                                     sync_list_base)

    # write voltage objects and property lists
    os.chdir(test_case_dir)
    with open('voltage_obj.txt', 'wb') as fp:
        pickle.dump(bus_list_voltage, fp)
    with open('voltage_prop.txt', 'wb') as fp:
        pickle.dump(prop_voltage, fp)

    # % load residential load data
    os.chdir(load_data_dir)
    # Check if current directory is empty
    if len(os.listdir(os.getcwd())) == 0:
        raise FileNotFoundError('No data files in directory: ', os.getcwd(), 'Please check the directory to '
                                                                             'ensure data file exists.')
    data_use = pandas.read_csv(base_load_file) # User can specify file within config.txt

    year = 2018     # NOTE: Make sure the year matches your data type (leap or not leap year) year or else the
# timestamps will be wrong and may throw errors.
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
    data_use = None  # clean up memory

    data_use_mat = np.asarray(data_use_filt[data_use_filt.columns[6:-1]]) * 1000
    agg_power = np.mean(data_use_mat, axis=1)
    admd = np.max(agg_power)
    print(f'Estimated admd from data is {admd}')
    admd = 3  # After diversity maximum demand.

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

    #   TODO: House Transformer configuration (NOTE: the nominal voltage should depend on the voltage at the node_name of
    #    the spot-load so a future task will be to automate this.
    #    For now, the code doesn't break because the voltages are the same everywhere

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

    ####  CONFIGURE LOAD OBJECTS AND TRANSFORMERS FOR DCFC SIMULATION STARTS HERE ####

    # Fast charging station transformer configuration.
    # Find only meters with ABC phases (3-phase) for DCFC connection.
    standard_rating = True  # use standard transformer ratings, not arbitrary
    glm_subset_dict_dcfc = {key: subdict for key, subdict in glm_dict_base.items() if
                            'name' in subdict.keys() and 'meter' in subdict['name'] and 'ABC' in subdict['phases']}
    num_fast_charging_nodes = param_dict['num_dcfc_nodes']
    num_charging_stalls_per_station = param_dict['num_dcfc_stalls_per_station']  # int
    charging_stall_base_rating = float(param_dict['dcfc_charging_stall_base_rating'].split('_')[0])  # kW
    trans_standard_ratings = np.array(
        [3, 6, 9, 15, 30, 37.5, 45, 75, 112.5, 150, 225, 300])  # units in kVA # TODO: get references for these
    DCFC_voltage = param_dict['dcfc_voltage']  # volts (480 volts is the most common DCFC transformer Secondary now)
    DCFC_trans_power_rating_kW = charging_stall_base_rating * num_charging_stalls_per_station  # kw base rating X number of stalls will oversize it for load
    load_pf = 0.95  # this can be >= and many EVSE can have pf close to 1. In initial simulation, they will be unity
    DCFC_trans_power_rating_kVA = DCFC_trans_power_rating_kW / load_pf
    proximal_std_rating = trans_standard_ratings[
        np.argmin(np.abs(trans_standard_ratings - DCFC_trans_power_rating_kVA))]  # find the closest transformer rating
    if standard_rating:
        DCFC_trans_power_rating_kVA = proximal_std_rating   # If using standard transformer sizes.
    charging_bus_subset_list = random.sample(list(glm_subset_dict_dcfc.values()), num_fast_charging_nodes)

    #   TODO: verify these or get more configs
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
        # TODO:
        # Not implemented yet, but we want to automate the creation of dedicated transformers if scenario is fast-charging
        pass

    k = 0

    fast_charging_bus_list = []
    # getting the 3-phase meters (nodes) that were stored in the charging_bus_subset_list
    for meter_dict in charging_bus_subset_list:
        # load object
        glm_house_dict[key_index] = {'name': 'dcfc_load_' + str(k),
                                     'load_class': 'C',
                                     'nominal_voltage': str(DCFC_voltage),
                                     'phases': 'ABCN',
                                     # this phase is currently...
                                     # ...hard-coded because we know we want to only connect to 3-phase connection
                                     'constant_power_A': '0.0+0.0j',
                                     'constant_power_B': '0.0+0.0j',
                                     'constant_power_C': '0.0+0.0j'}  # the powers get populated in simulation

        # need to check the phase of the load object to determine how to choose the transformer phase

        obj_type[key_index] = {'object': 'load'}
        key_index = key_index + 1

        #   Transformer (DCFC transformer) setup.
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
    np.savetxt('dcfc_bus.txt', fast_charging_bus_list, fmt="%s")  # Store all the nodes in which there is DCFC.

    ####  CONFIGURE LOAD OBJECTS AND TRANSFORMERS FOR DCFC SIMULATION ENDS HERE ####

    ######### L2 SETUP STARTS HERE ###########
    #   This is the transformer configuration that is inherited for L2 208-240V
    standard_rating = True  # TODO: include this in the config.txt options.
    glm_subset_dict_L2 = {key: subdict for key, subdict in glm_dict_base.items() if
                          'name' in subdict.keys() and 'meter' in subdict['name'] and 'ABC' in subdict['phases']}
    num_L2_charging_nodes = param_dict['num_l2_nodes']
    num_charging_stalls_per_station = param_dict['num_l2_stalls_per_station']  # make param later
    charging_stall_base_rating = float(
        param_dict['l2_charging_stall_base_rating'].split('_')[0])  # kW (make param later)
    L2_voltage = param_dict['l2_voltage']  # Volts (usually 208 - 240V)
    L2_trans_power_rating_kW = charging_stall_base_rating * num_charging_stalls_per_station  # kw
    load_pf = 0.95  # this can be >= and many EVSE can have pf close to 1. In initial simulation, they will be unity
    L2_trans_power_rating_kVA = L2_trans_power_rating_kW / load_pf
    proximal_std_rating = trans_standard_ratings[
        np.argmin(np.abs(trans_standard_ratings - L2_trans_power_rating_kVA))]  # find the closest transformer rating
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
    fraction_commercial_sec_node = 0.3  # Fraction of nodes to be assigned. Changeable.
    # CALCULATE LOAD MAGNITUDE FOR EACH SPOT LOAD
    spot_load_magnitude = [abs(spot_load_list[i]) / 1000 for i in range(len(spot_load_list))]
    commercial_load_indices = []
    contains_commercial_load = []
    if num_L2_charging_nodes > 0:
        commercial_load_indices = [i for i in range(len(spot_load_list)) if
                                   spot_load_magnitude[i] > L2_trans_power_rating_kVA]
        contains_commercial_load = random.sample(commercial_load_indices,
                                                 max(int(fraction_commercial_sec_node * len(commercial_load_indices)),
                                                     1))

    # select (sample) triplex node_name that will have L2 charging in addition to commercial load (e.g. work buildings/hotels)
    L2_charging_node_options = []

    k = 0
    real_power_df, reactive_power_df = None, None
    for i in range(len(bus_list)):
        commercial_load = False
        if i in contains_commercial_load:
            commercial_load = True
            num_transformers = int((abs(spot_load_list[i]) / (
                    L2_trans_power_rating_kVA * 1000)))  # np.floor because of higher kVA leads to no transformers and causes downstream errors
        else:
            num_transformers = int((abs(spot_load_list[i]) / (
                    20 * 1000)))  # todo: 20 was used here because of the tranformer rating is 20
        num_transformers_list.append(num_transformers)
        for j in range(num_transformers):  # number of transformers per bus
            # Triplex node_name
            if commercial_load:
                # todo: is there a way to initially set loading?
                num_houses = int((L2_trans_power_rating_kVA * pf_min) * safety_factor / admd)
            else:
                num_houses = int((20 * pf_min) * safety_factor / admd)  # admd is the max power. 0.85 is the worst pf
            real_power_trans = np.sum(
                data_use_mat[:, np.random.choice(np.arange(data_use_mat.shape[1]), size=(num_houses,))], axis=1)
            pf_trans = np.random.uniform(pf_min, pf_max, size=real_power_trans.shape)  # sample power factor
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
                                             'constant_power_12': str(spot_load_list[i] / (num_transformers + 3)).replace('(',
                                                                                                                 '').replace(
                                                 ')',
                                                 '')}  # +3 is an ad-hoc way to make base case run normally
                L2_charging_node_options += f'tn_{k}',  # add this into options for L2 charging site for sim
            else:
                glm_house_dict[key_index] = {'name': 'tn_' + str(k),
                                             'nominal_voltage': '120.00',
                                             'phases': str(load_phases[i]) + "S",
                                             'constant_power_12': str(spot_load_list[i] / (num_transformers + 3)).replace('(',
                                                                                                                 '').replace(
                                                 ')',
                                                 '')}  # +3 is an ad-hoc way to make base case run normally
            obj_type[key_index] = {'object': 'triplex_load'}
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

    # Now get the desired L2 charging locs and save them for simulation
    L2_charging_bus_subset_list = random.sample(L2_charging_node_options, num_L2_charging_nodes)
    os.chdir(test_case_dir)
    np.savetxt('L2charging_bus.txt', L2_charging_bus_subset_list, fmt="%s")

    # write out glm file for secondary distribution
    out_dir = test_case_dir
    file_name = feeder_name + '_secondary.glm'
    glm_mod_functions.write_base_glm(glm_house_dict, obj_type, globals_list, include_list, out_dir, file_name,
                                     sync_list)

    # save load data (THESE ARE TYPICALLY UNCONTROLLABLE LOADS)
    os.chdir(test_case_dir)
    real_power_df.to_csv('real_power.csv', index=False)
    reactive_power_df.to_csv('reactive_power.csv', index=False)


if __name__ == "__main__":
    main()
