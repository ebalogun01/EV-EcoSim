#%%

import glm_mod_functions
import os
import pandas
import datetime
import sklearn.preprocessing
import numpy as np
import ast
import pickle

#read config file
f=open('config.txt','r')
param_dict=f.read()
f.close()
param_dict=ast.literal_eval(param_dict)

feeder_name=param_dict['feeder_name']
set_sd=param_dict['set_sd']
mean_scale=param_dict['mean_scale']
base_file_dir=param_dict['base_file_dir']
player_data_dir=param_dict['player_data_dir']
load_data_dir=param_dict['load_data_dir']
box_pts=param_dict['box_pts']
starttime_str=param_dict['starttime']
endtime_str=param_dict['endtime']

base_glm_file=feeder_name+'.glm'
print('Loading original glm')
glm_dict_base,obj_type_base,globals_list_base,include_list_base,sync_list_base=glm_mod_functions.load_base_glm(base_file_dir,base_glm_file)


for i in obj_type_base.keys():
    if ('object' in obj_type_base[i].keys()): 
        
        #modify load objects
        if 'load' in obj_type_base[i]['object']:
            
            
            