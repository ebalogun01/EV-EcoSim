import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

#define global python simulation variables and default initial values

#iteration number
it=0

#power flow timestep
pf_dt=60

#uncontrollable load profiles

p_df=pd.read_csv('real_power.csv')
q_df=pd.read_csv('reactive_power.csv')

p_array=np.asarray(p_df)
q_array=np.asarray(q_df)
#p_array=s_array*pf_array
#q_array=(s_array**2-p_array**2)**0.5
x_array=np.concatenate((p_array,q_array),axis=1)
scaler=StandardScaler()
scaler.fit(x_array)
x_array_scaled=scaler.transform(x_array)
x_array_scaled=np.concatenate((np.ones((x_array_scaled.shape[0],1)),x_array_scaled),axis=1)
x_array_aug=np.concatenate((np.ones((x_array.shape[0],1)),x_array),axis=1)

# voltage objects and properties

with open('voltage_obj.txt','rb') as fp:
    voltage_obj=pickle.load(fp)
with open('voltage_prop.txt','rb') as fp:
    voltage_prop=pickle.load(fp)
vm=np.zeros((1,len(voltage_obj)))
vp=np.zeros((1,len(voltage_obj)))
v_pred=np.zeros((1,len(voltage_obj)))


# initialize RLSF parameters

lam=0.99
N=x_array_scaled.shape[1]
#weight matrix, shape: (num powers,num voltages)
w=np.random.random((x_array_scaled.shape[1],int(len(voltage_obj))*1))
Q=np.eye(N)
rmse_vmag=[]
