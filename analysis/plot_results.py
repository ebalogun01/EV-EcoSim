"""This file is used post-simulation to parse the voltages from the power-simulation to calculate the percentage
voltage violations per ANSI C84.1. The file also generates voltage distribution plots. A user can modify the
SIMULATION_FOLDER variable which is the string of the path where the powerflow simulation output voltages at each node
exist.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


SIMULATION_FOLDER = '../charging_sim/3200evs/ah_scaling/voltparse'
os.chdir(SIMULATION_FOLDER)
FIG_WIDTH, FIG_HEIGHT = 10, 6
FONT_SIZE = 14
DAY_MINUTES = 1440
NUM_DAYS = 30   # can change this depending on the desired number of days
end_idx = DAY_MINUTES * NUM_DAYS
plt.rcParams.update({'font.size': FONT_SIZE})

for root, dirs, files, in os.walk(".", topdown=True):
    for name in dirs:
        curr_dir = os.getcwd()
        os.chdir(os.path.join(root, name))
        curr_path = os.path.join(root, name)
        print(os.getcwd())
        voltages = pd.read_csv('voltages.csv', on_bad_lines='skip')[:end_idx]  # read this as sparse or eliminate nodes with 0 values for V
        # above includes patch for skipping problematic lines, but needs to be fixed in voltdump2.py later
        plt.close('all')

        # process voltages
        cols = list(voltages.columns)
        real_cols_ind = [(('_Ar' in m) or ('_Br' in m) or ('_Cr' in m)) for m in cols]
        imag_cols_ind = [(('_Ai' in m) or ('_Bi' in m) or ('_Ci' in m)) for m in cols]
        real_cols = [x for x, y in zip(cols, real_cols_ind) if y == True]
        imag_cols = [x for x, y in zip(cols, imag_cols_ind) if y == True]

        voltage_real = voltages.loc[:, real_cols_ind]
        voltage_imag = voltages.loc[:, imag_cols_ind]

        voltage_real = np.asarray(voltage_real)
        voltage_imag = np.asarray(voltage_imag)

        # calculate complex voltage, mag, phase
        voltage_complex = voltage_real + voltage_imag * 1j
        v = abs(voltage_complex)
        a = np.rad2deg(np.angle(voltage_complex))

        a[a < 50] = a[a < 50] + 120     # need to check these again
        a[a > 50] = a[a > 50] - 120

        mean_v = np.mean(v, axis=0, keepdims=True)
        zero_idx = np.argwhere(mean_v[..., :] == 0)
        mean_v = np.delete(mean_v, zero_idx, axis=1)  # remove zero voltages
        v = np.delete(v, zero_idx, axis=1)  # remove zero voltages from data

        nom_v = np.zeros(mean_v.shape)

        nom_v[(mean_v < 300) * (mean_v > 250)] = 480 / (3 ** 0.5)
        nom_v[(mean_v < 8000) * (mean_v > 6500)] = 7200
        nom_v[(mean_v < 2600) * (mean_v > 2100)] = 2401.7
        nom_v[(mean_v < 150) * (mean_v > 80)] = 120
        nom_v[(mean_v < 500) * (mean_v > 460)] = 480
        nom_v[(mean_v < 250) * (mean_v > 230)] = 240
        norm_v = v / nom_v.reshape(1, -1)

        # calculating percentage of voltage violations
        num_voltage_violations = len(norm_v.flatten()[(norm_v.flatten() > 1.05) | (norm_v.flatten() < 0.95)])
        percent_violations = num_voltage_violations/len(norm_v.flatten()) * 100
        np.savetxt('percent_voltage_violations.csv', [percent_violations])

        print(f'Dim v: {str(v.shape)}')
        print(
            f'Dim 480 V: {str(len(mean_v[(mean_v < 300) * (mean_v > 250)]) + len(mean_v[(mean_v < 500) * (mean_v > 460)]))}')
        print(f'Dim 7200 V: {len(mean_v[(mean_v < 8000) * (mean_v > 6500)])}')
        print(f'Dim 120 V: {len(mean_v[(mean_v < 150) * (mean_v > 80)])}')
        print(f'Dim 2400 V: {len(mean_v[(mean_v < 2600) * (mean_v > 2100)])}')
        print(f'Dim 240 V: {len(mean_v[(mean_v < 250) * (mean_v > 230)])}')

        # calculate percentage violations
        print("Printing shapes for sanity check...")
        print(norm_v.shape)
        print(v.shape)
        print(a.shape)
        print("Done printing shapes.")

        print("Creating voltage plots...")
        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        for i in range(v.shape[1]):
            plt.plot(v[:, i] / 1000)

        plt.ylabel('Voltage (kV)')
        plt.xlabel('Time (min)')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        plt.grid()
        plt.tight_layout()
        plt.savefig('v.png')
        plt.close()

        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        for i in range(v.shape[1]):
            plt.plot(norm_v[:, i])
        plt.ylabel('Voltage (p.u.)')
        plt.xlabel('Time (min)')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        plt.grid()
        plt.tight_layout()
        plt.savefig('vnorm.png')
        plt.close()

        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.hist(np.ndarray.flatten(norm_v), bins=100, range=[0.85, 1.15])
        plt.ylabel('Count', fontsize=16)
        plt.xlabel('Voltage (p.u.)', fontsize=16)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=16)
        plt.xlim([0.85, 1.15])
        plt.grid()
        plt.tight_layout()
        plt.savefig('vhist.png')
        plt.close()

        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
        plt.hist(np.ndarray.flatten(a), bins=100, range=[-15, 5])
        plt.ylabel('Count', fontsize=16)
        plt.xlabel('Voltage phase (deg)')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        plt.grid()
        plt.tight_layout()
        plt.savefig('ahist.png')
        plt.close()
        print("Finished creating voltage plots.")

        os.chdir("..")
