"""@author: Emmanuel Balogun. Testing file for E.Balogun. Please ignore"""

import matplotlib.pyplot as plt
import numpy as np
from battery import Battery
import scipy.io
from batteryAgingSim import BatterySim
import json

# TODO: load the battery and simulate it using the load profiles from Simona data
# TODO: plot the aging path to see how it goes...perform comparisons
num_steps = 1
file = '../battery_data/W9_cycle.mat'


#   LOAD BATTERY CONFIG
with open('configs/battery.json', 'r') as f:
    c = json.load(f)
    c['cell_nominal_cap'] = c['capacity']
    c['resolution'] = 1/60  # one second

params_list = [key for key in c if "params_" in key]
for params_key in params_list:
    c[params_key] = np.loadtxt(f'../{c[params_key]}')  # replace path with true value
# do the OCV maps as well; reverse directionality is important for numpy.interp function
c["OCV_map_voltage"] = np.loadtxt(f'../{c["OCV_map_voltage"]}')[::-1]
c["OCV_map_SOC"] = np.loadtxt(f'../{c["OCV_map_SOC"]}')[::-1]
c['SOC'] = 1e-8


# INITIALIZE MODULES
battery = Battery(config=c)
aging = BatterySim(0, num_steps=1, res=1/60)

current = scipy.io.loadmat(file)['I_full_vec_M1_NMC25degC']
for i in range(current.shape[0]):
    battery.dynamics(current[i, 0])
    if i > 0:
        aging.run(battery)
        print('Second: ', i, 'Current: ', current[i, 0], 'Capacity: ', battery.cap)
    if battery.cap < 0:
        print('Error')
        break

# WHEN DONE PLOT STUFF
print("Battery Capacity is: ", battery.cap)
plt.plot(battery.SOH_track)
plt.xlabel('Time-step')
plt.ylabel('SOH')
plt.show()
np.savetxt('Capacity_verification.csv', np.array(battery.SOH_track) * c['capacity'])

c_int = np.cumsum(current)
cycles = c_int / (3600 * c['capacity'] * 2)