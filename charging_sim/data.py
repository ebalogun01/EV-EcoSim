import numpy as np
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

mat = loadmat('Datasets/data_NMC25degC.mat')  # load mat-file
raw_data_NMC25degC = {}
current_list = [5, 10, 15, 30]
# obtain voltages from data @ 25C
V_NMC_M1_25C = mat['V_Dis_M1_NMC25degC']
OCV_NMC_M1_25C = V_NMC_M1_25C[:, 0]
OCV_NMC_M1_25C = OCV_NMC_M1_25C[~np.isnan(OCV_NMC_M1_25C)]
len_desired = 300


def reduce_vector(vector, desired_len):
    vector_length = len(vector)
    step_size = 1
    if vector_length > desired_len:
        dt = vector_length/desired_len
        step_size = int(dt)
    new_vector = vector[0::step_size]
    new_length = len(new_vector)
    if new_length > desired_len:
        new_vector = new_vector[0:desired_len]
    new_vector.shape = (desired_len, 1)
    return new_vector, dt


def clean_vector(vec):
    vec = vec[~np.isnan(vec)]
    vec = vec[vec != 0]
    return vec


def get_OCV(SoC):
    global OCV_NMC_M1_25C
    SOC_OCV_NMC_M1_25C = clean_vector(SOC_NMC_M1_25C[:, 0])
    last_index = min(len(OCV_NMC_M1_25C), len(SOC_OCV_NMC_M1_25C))
    SOC_OCV_NMC_M1_25C = SOC_OCV_NMC_M1_25C[0:last_index]
    OCV_NMC_M1_25C = OCV_NMC_M1_25C[0:last_index]
    # print(OCV_NMC_M1_25C.shape, SOC.shape, SOC_OCV_NMC_M1_25C.shape)
    return np.interp(SoC, SOC_OCV_NMC_M1_25C[::-1], OCV_NMC_M1_25C[::-1])

def get_SOC(ocv):
    global OCV_NMC_M1_25C
    SOC_OCV_NMC_M1_25C = clean_vector(SOC_NMC_M1_25C[:, 0])
    last_index = min(len(OCV_NMC_M1_25C), len(SOC_OCV_NMC_M1_25C))
    SOC_OCV_NMC_M1_25C = SOC_OCV_NMC_M1_25C[0:last_index]
    OCV_NMC_M1_25C = OCV_NMC_M1_25C[0:last_index]
    # print(OCV_NMC_M1_25C.shape, SOC.shape, SOC_OCV_NMC_M1_25C.shape)
    return np.interp(ocv, OCV_NMC_M1_25C[::-1], SOC_OCV_NMC_M1_25C[::-1])



V_1C = V_NMC_M1_25C[:, 1]
V_2C = V_NMC_M1_25C[:, 2]
V_3C = V_NMC_M1_25C[:, 3]
V_5C = V_NMC_M1_25C[:, 4]

V_1C, dt_1 = reduce_vector(clean_vector(V_1C), len_desired)
V_2C, dt_2 = reduce_vector(clean_vector(V_2C), len_desired)
V_3C, dt_3 = reduce_vector(clean_vector(V_3C), len_desired)
V_5C, dt_5 = reduce_vector(clean_vector(V_5C), len_desired)


SOC_NMC_M1_25C = mat['SOC_dis_M1_NMC25degC']
SOC_OCV = SOC_NMC_M1_25C[:, 0]
SOC_1C = SOC_NMC_M1_25C[:, 1]
SOC_2C = SOC_NMC_M1_25C[:, 2]
SOC_3C = SOC_NMC_M1_25C[:, 3]
SOC_5C = SOC_NMC_M1_25C[:, 4]

SOC_OCV = clean_vector(SOC_OCV)
SOC_1C = clean_vector(SOC_1C)
SOC_2C = clean_vector(SOC_2C)
SOC_3C = clean_vector(SOC_3C)
SOC_5C = clean_vector(SOC_5C)


# Truncate the vectors to reduce to desired length X 1 vectors
SOC_OCV, dt_OCV = reduce_vector(SOC_OCV, len_desired)
OCV, dt_OCV = reduce_vector(OCV_NMC_M1_25C, len_desired)
raw_data_NMC25degC['OCV'] = np.concatenate((SOC_OCV, OCV), axis=1)
# plot = raw_data_NMC25degC['OCV'][0:len_desired-1, 1] - raw_data_NMC25degC['OCV'][1:, 1]
# plt.plot((plot))
# plt.show()

SOC_1C, dt_1b = reduce_vector(SOC_1C, len_desired)
print(SOC_1C.shape)
OCV_1C = get_OCV(SOC_1C)
raw_data_NMC25degC['1C'] = np.concatenate((SOC_1C, V_1C), axis=1)
# plot = (raw_data_NMC25degC['1C'][0:len_desired-1, 1] - raw_data_NMC25degC['1C'][1:, 1])/current_list[0]
# plt.plot((plot))
# plt.show()
OCV_NMC = get_OCV(np.load('BatteryData/SOC_INR21700_T25_Fast_Pulse_Dis_1C_X1_Channel_2.npy'))
np.save('BatteryData/OCV_INR21700_T25_Fast_Pulse_Dis_1C_X1_Channel_2.npy', OCV_NMC)

SOC_2C, dt_2b = reduce_vector(SOC_2C, len_desired)
OCV_2C = get_OCV(SOC_2C)
raw_data_NMC25degC['2C'] = np.concatenate((SOC_2C, V_2C), axis=1)
# plot = (raw_data_NMC25degC['2C'][0:len_desired-1, 1] - raw_data_NMC25degC['2C'][1:, 1])/current_list[1]
# plt.plot((plot))
# plt.show()


SOC_3C, dt_3b = reduce_vector(SOC_3C, len_desired)
OCV_3C = get_OCV(SOC_3C)    # This is not OCV but just discharge voltage at C-rate so var name is a misnomer
raw_data_NMC25degC['3C'] = np.concatenate((SOC_3C, V_3C), axis=1)
# plot = (raw_data_NMC25degC['3C'][0:len_desired-1, 1] - raw_data_NMC25degC['3C'][1:, 1])/current_list[2]
# plt.plot((plot))
# plt.show()


SOC_5C, dt_5b = reduce_vector(SOC_5C, len_desired)
OCV_5C = get_OCV(SOC_5C)
raw_data_NMC25degC['5C'] = np.concatenate((SOC_5C, V_5C), axis=1)
# plot = (raw_data_NMC25degC['5C'][0:len_desired-1, 1] - raw_data_NMC25degC['5C'][1:, 1])/current_list[3]
# plt.plot((plot))
# plt.show()

# plt.legend(["0C", "1C", "2C", "3C", "5C"])
# plt.show()

SOC_list = [SOC_1C, SOC_2C, SOC_3C, SOC_5C]
OCV_list = [OCV_1C, OCV_2C, OCV_3C, OCV_5C]
Volt_list = [V_1C, V_2C, V_3C, V_5C]
dt_list = [dt_1, dt_2, dt_3, dt_5]
dt_list2 = [dt_1b, dt_2b, dt_3b, dt_5b]

# fit OCV model of form OCV = M * SOC + CONSTANT
reg = LinearRegression().fit(SOC_OCV, OCV)
OCV_SOC_linear_params = np.vstack([reg.coef_, reg.intercept_])
np.save('BatteryData/OCV_SOC_linear_params_NMC_25degC.npy', OCV_SOC_linear_params) # Saves this to file
SOC = SOC_OCV

plt.plot(SOC, reg.predict(SOC))
plt.xlim(max(SOC), 0)
plt.plot(SOC, OCV)
plt.xlabel("State of Charge")
plt.ylabel("Open Circuit Voltage")

plt.show()
# plt.plot(SOC_NMC_M1_25C, OCV_NMC_M1_25C)
# plt.show()

# SOC_charge = SOC_charge/max(SOC_charge)
# SOC_charge = df['Charge_Capacity(Ah)']
# SOC_charge = SOC_charge/4.85
# SOC_charge = SOC_charge[0:14788]
# SOC_discharge = df['Charge_Capacity(Ah)']
# SOC_discharge = df['Discharge_Capacity(Ah)']
# SOC_discharge = SOC_discharge[14788:]
# SOC_discharge = 4.85 - SOC_discharge
#
# SOC_NMC_25degC = np.concatenate([SOC_charge, SOC_discharge], axis=0)
# SOC_discharge = df['Discharge_Capacity(Ah)']
# SOC_discharge = 4.85 - SOC_discharge
# SOC_discharge = SOC_discharge/4.85
# SOC_discharge = df['Discharge_Capacity(Ah)']
# SOC_discharge = (4.85 - SOC_discharge)/4.85
# SOC_discharge = SOC_discharge[14788:]
# SOC_NMC_25degC = np.concatenate([SOC_charge, SOC_discharge], axis=1)
#
# SOC_NMC_25degC = np.concatenate([SOC_charge, SOC_discharge], axis=0).T
# SOC_NMC_25degC = np.reshape(SOC_NMC_25degC, (SOC_NMC_25degC.size, 1))
# np.save('SOC_INR21700_T25_Fast_Pulse_Dis_1C_X1_Channel_2.npy', SOC_NMC_25degC)
