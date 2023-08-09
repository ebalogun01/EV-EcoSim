import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import sys
import os
#%%
print(os.getcwd())
# sys.path.append('../../../EV50_cosimulation/test_cases/battery')
exp_path = 'test_cases/battery/'
#%%
# data_files = ['XX21', 'ZX21', 'YY21', 'ZY21']
data_files = ['June0', 'June24', 'June3', 'June4']

#%% get the tranformer loads
trans_loading_data_1 = pd.read_csv(f'{exp_path}test_{data_files[0]}/trans_loading_percent.csv')
dcfc_trans_loading_data_1 = trans_loading_data_1["dcfc_trans_0"].values.reshape(-1, 1)
trans_tn_loading_1 = trans_loading_data_1["trip_trans_72"].values.reshape(-1, 1)

trans_loading_data_2 = pd.read_csv(f'{exp_path}test_{data_files[1]}/trans_loading_percent.csv')
dcfc_trans_loading_data_2 = trans_loading_data_2["dcfc_trans_0"].values.reshape(-1, 1)
trans_tn_loading_2 = trans_loading_data_2["trip_trans_72"].values.reshape(-1, 1)

# dcfc_trans_th_data_1 = pd.read_csv(f'test_{data_files[0]}')
# dcfc_trans_th_data_2 = pd.read_csv(f'test_{data_files[0]}')

patterns = ['o', '/', '///', '\\', 'x', '.', '*']
#%%     Get the transformer temperatures
trans_tn_temperature_data_1 = pd.read_csv(f'{exp_path}test_{data_files[0]}/trans_th.csv')
trans_tn_temperature_data_2 = pd.read_csv(f'{exp_path}test_{data_files[1]}/trans_th.csv')

dcfc_trans_th_data_1 = trans_tn_temperature_data_1['dcfc_trans_0'].values.reshape(-1, 1)
tn_trans_th_data_1 = trans_tn_temperature_data_1["trip_trans_72"].values.reshape(-1, 1)

dcfc_trans_th_data_2 = trans_tn_temperature_data_2['dcfc_trans_0'].values.reshape(-1, 1)
tn_trans_th_data_2 = trans_tn_temperature_data_2["trip_trans_72"].values.reshape(-1, 1)



#%%     Start plots here

# loading percentage
# x_vals = np.arange(1, 25, 1)

def plot_trans_loading(y, x=None, hours=True, onemin=True, fifteenmin=False):
    # todo: using x as input
    if onemin:
        x_vals = 1/60 * np.arange(0, y.shape[0])
    elif fifteenmin:
        x_vals = 15/60 * np.arange(0, 1440)
    plot_rows, plot_cols = 2, 2
    fig, axs = plt.subplots(plot_rows, plot_cols)
    y_idx = 0
    x_ticks = list(range(0, int(round(max(x_vals)))+1, 3))
    for i in range(plot_rows):
        for j in range(plot_cols):
            axs[i, j].plot_tables(x_vals, y[:, y_idx], color='b')
            axs[i, j].set_xlim(left=0, right=24)
            # axs[i, j].set_xticklabels(x_ticks)
            # axs[i, j].set_ylabel("Transformer loading (%)")
            axs[i, j].set_xlabel("Hour of day")
            y_idx += 1
    fig.supylabel("Transformer loading (%)")
    fig.tight_layout()
    plt.show()


def plot_trans_temp(y, x=None, hours=True, onemin=True, fifteenmin=False):
    # todo: using x as input
    if onemin:
        x_vals = 1 / 60 * np.arange(0, y.shape[0])
    if fifteenmin:
        x_vals = 15 / 60 * np.arange(0, 1440)
    plot_rows, plot_cols = 2, 2
    fig, axs = plt.subplots(plot_rows, plot_cols)
    y_idx = 0
    # x_ticks = list(range(1, int(round(max(x_vals)))+1, 4))
    for i in range(plot_rows):
        for j in range(plot_cols):
            axs[i, j].plot_tables(x_vals, y[:, y_idx], color='r')
            axs[i, j].set_xlim(left=0, right=24)
            # axs[i, j].set_xticks(x_ticks)
            # axs[i, j].set_xticklabels(x_ticks)
            # axs[i, j].set_ylabel("Transformer Temperature ($^\circ$ C)")
            axs[i, j].set_xlabel("Hour of day")
            y_idx += 1
    fig.supylabel("Transformer Temperature ($^\circ$C)")
    # plt.ylabel("Transformer Temperature ($^\circ$C)")
    fig.tight_layout()
    plt.show()


#%%
trans_temps = np.hstack((dcfc_trans_th_data_1, tn_trans_th_data_1, dcfc_trans_th_data_2, tn_trans_th_data_2))
trans_loading = np.hstack((dcfc_trans_loading_data_1, trans_tn_loading_1, dcfc_trans_loading_data_2, trans_tn_loading_2))

#%%
plot_trans_temp(trans_temps[1:1441, ])  # always start from one since temp depends on loading so loading is cardinality - 1
plot_trans_loading(trans_loading[:1440, ])




