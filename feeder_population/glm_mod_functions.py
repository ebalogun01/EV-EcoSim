import os

"""
Functions for modifying GridLAB-D power system simulation models
"""

def load_base_glm(base_file_dir,base_glm_file):
    """This loads the glm as list and populates it into dict and returns the output"""
    
    os.chdir(base_file_dir)
    f = open(base_glm_file, 'r')
    glm=f.readlines()
    
    glm_dict={}
    obj_type={}
    glm_list=list()
    globals_list=list()
    include_list=list()
    sync_list=list()
    
    #edit out all comments
    for l in glm:
        line_temp=l.lstrip().rstrip().rstrip('\n').split('//')[0]       # is the // encoding something
        #Comment somewhere in line
        if len(line_temp)>1:
            #There is some content in line, extract content
            if l.split('//')[0]!='':
                glm_list.append(line_temp.rstrip())     # this adds to glm list lines that are not comments
        #No comment in line
        else:
            #Line is not a space
            if line_temp!='':
                glm_list.append(line_temp)
                
    obj_num=0
    obj_flag=0
    #put info into dict structure
    for l in glm_list:
        #Setting global variable
        if l[0:4]=='#set':
            globals_list.append(l)
        elif l[0:8]=='#include':
            include_list.append(l)
        elif 'object' in l:
            obj_flag=1
            line_temp=l.rstrip('{').rstrip().split(' ')
            obj_type[obj_num]={'object':line_temp[1]}
            prop_num=0
        elif ('class' in l) and obj_flag==0:
            obj_flag=1
            line_temp=l.rstrip('{').rstrip().split(' ')
            obj_type[obj_num]={'class':line_temp[1]}
            prop_num=0            
        elif 'module' in l:
            obj_flag=1
            line_temp=l.rstrip('{').rstrip().split(' ')
            #if no properties in object
            if ';' in line_temp[1]:
                obj_type[obj_num]={'module':line_temp[1].rstrip(';')}
                obj_flag=0
                glm_dict[obj_num]={}
                obj_num=obj_num+1
            #if properties in object
            else:
                obj_type[obj_num]={'module':line_temp[1]}
                obj_flag=1
                prop_num=0
        elif 'clock' in l:
            obj_flag=1
            obj_type[obj_num]={'clock':'clock'}
            prop_num=0
        elif 'script' in l:
            sync_list.append(l)
        
        
        elif l=='}' or l=='};':
            obj_num=obj_num+1
            obj_flag=0
        else:
            if obj_flag==1:
                line_temp=l.split(' ',maxsplit=1)
                if prop_num==0:
                    glm_dict[obj_num]={line_temp[0]:line_temp[1].rstrip(';')}
                    prop_num=prop_num+1
                else:
                    glm_dict[obj_num][line_temp[0]]=line_temp[1].rstrip(';')   
            else:
                print('error')
                print(l)
    return glm_dict,obj_type,globals_list,include_list,sync_list

def write_base_glm(glm_dict,obj_type,globals_list,include_list,out_dir,file_name,sync_list):
    os.chdir(out_dir)    
    glm_out = open(file_name,"w+")
    
    
    for i in range(len(globals_list)):
        glm_out.write(globals_list[i]+'\n\n')
        
    for i in glm_dict.keys():    
    #for i in range(len(glm_dict)):
        if 'clock' in obj_type[i].keys():
            write_clock_dict(glm_out,glm_dict[i])
        
    for i in glm_dict.keys():    
    #for i in range(len(glm_dict)):
        if 'module' in obj_type[i].keys():
            write_mod_dict(glm_out,glm_dict[i],obj_type[i]['module'])
    
    for i in range(len(include_list)):
        glm_out.write(include_list[i]+'\n\n')

    for i in glm_dict.keys():
        if 'filter' in obj_type[i].keys():
            write_filter_dict(glm_out,glm_dict[i],obj_type[i]['filter'])
    
    for i in glm_dict.keys():
        if 'class' in obj_type[i].keys():
            write_class_dict(glm_out,glm_dict[i],obj_type[i]['class'])
    
    for i in glm_dict.keys():    
    #for i in range(len(glm_dict)):
        if 'object' in obj_type[i].keys():
            if 'player' in obj_type[i]['object']:
                write_obj_dict(glm_out,glm_dict,i,obj_type[i]['object'])
            
    for i in glm_dict.keys():    
    #for i in range(len(glm_dict)):
        if 'object' in obj_type[i].keys():
            if ('player' in obj_type[i]['object'])==False:
                write_obj_dict(glm_out,glm_dict,i,obj_type[i]['object'])

    for i in range(len(sync_list)):
        glm_out.write(sync_list[i]+'\n\n')
    
    glm_out.close()

def write_obj_dict(file,gld_dict,dict_key,obj_type):
    '''Write dictionary corresponding to GLD objects to .glm file'''
    
    if dict_key==-1:
        file.write('object '+obj_type+' {\n')
        for i,j in gld_dict.items():
            file.write('\t'+str(i)+' '+str(j)+';\n')
        file.write('}\n\n')
    else:
        file.write('object '+obj_type+' {\n')
        for i,j in gld_dict[dict_key].items():
            file.write('\t'+str(i)+' '+str(j)+';\n')
        file.write('}\n\n')
        
        
def write_mod_dict(file,gld_dict,mod_name):
    '''Write dictionary corresponding to GLD module to .glm file'''
    if len(gld_dict)==0:
        file.write('module '+mod_name+';\n\n')
    else:
        file.write('module '+mod_name+' {\n')
        for i,j in gld_dict.items():
            file.write('\t'+str(i)+' '+str(j)+';\n')
        file.write('}\n\n')

def write_class_dict(file,gld_dict,class_name):
    '''Write dictionary corresponding to GLD class to .glm file'''
    if len(gld_dict)==0:
        file.write('class '+class_name+';\n\n')
    else:
        file.write('class '+class_name+' {\n')
        for i,j in gld_dict.items():
            file.write('\t'+str(i)+' '+str(j)+';\n')
        file.write('}\n\n')

def write_filter_dict(file,gld_dict,class_name):
    '''Write dictionary corresponding to GLD filter to .glm file'''
    if len(gld_dict)==0:
        file.write('filter '+class_name+';\n\n')
    else:
        file.write('filter '+class_name+' {\n')
        for i,j in gld_dict.items():
            file.write('\t'+str(i)+' '+str(j)+';\n')
        file.write('}\n\n')
            
def write_clock_dict(file,gld_dict):
    '''Write dictionary corresponding to GLD clock to .glm file'''
    
    file.write('clock {\n')
    for i,j in gld_dict.items():
        file.write('\t'+str(i)+' '+str(j)+';\n')
    file.write('}\n\n')


def replace_load_w_meter_old(glm_dict,match_str,rep_str,obj_type):
    '''Replace all instances of property in glm_dict'''
    replace_prop_list=list()
    for i in glm_dict.keys():
        if match_str in glm_dict[i].values():
            replace_prop_list.append(i)
    for i in replace_prop_list:
        for prop in glm_dict[i].keys():
            if glm_dict[i][prop]==match_str:
                glm_dict[i][prop]=rep_str
            if obj_type[i]['object']=='load' and prop[0:8]=='constant':
                delete_index=i
                delete_prop=prop
                
        if obj_type[i]['object']=='load':
            obj_type[i]['object']='meter'
    del glm_dict[delete_index][delete_prop]            
    return glm_dict

def replace_load_w_meter(glm_dict,match_str,rep_str,obj_type):
    '''Replace all instances of property in glm_dict'''
    replace_prop_list=list()
    for i in glm_dict.keys():
        if match_str in glm_dict[i].values():
            replace_prop_list.append(i)
    delete_index_list=[]
    delete_prop_list=[]
    for i in replace_prop_list:
        for prop in glm_dict[i].keys():
            if glm_dict[i][prop]==match_str:
                glm_dict[i][prop]=rep_str
            if obj_type[i]['object']=='load' and prop[0:8]=='constant':
                delete_index_list.append(i)
                delete_prop_list.append(prop)
                
        if obj_type[i]['object']=='load':
            obj_type[i]['object']='meter'
    for i in range(len(delete_index_list)):
        del glm_dict[delete_index_list[i]][delete_prop_list[i]]            
    return glm_dict