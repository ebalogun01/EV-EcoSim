import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
# import plotly.io as pio

# Specify if we would like to load existing data or pull fresh data
load_existing_data = True
if not load_existing_data:
    from data import raw_data_NMC25degC

# from rawData import raw_data_LiNiMnCoO2
# import matplotlib as cm
# from mpl_toolkits.mplot3d import Axes3D
# import plotly.express as px
# import plotly.graph_objects as go

# pio.renderers.default = "browser"
# SOC = np.linspace(0, 1, 25)
# amps = np.linspace(0, 8, 5)
# temp = np.linspace(0, 50, 6)


class BatteryMaps:

    """Class holds data Models for battery behavior."""

    def __init__(self, raw_data=None):
        self.response_volume = None
        self.response_surface = None
        self.raw_data = raw_data

    def get_response_volume(self):
        """Uses battery Data to create response map with input: T/C, SOC, Current/A. Out: Mapping Function"""
        power_3D, voltage_3D, temp_SOC_amp_x, temp_SOC_amp_y, temp_SOC_amp_z = self.process_data(self.raw_data)
        self.response_volume = RegularGridInterpolator((temp, amps, SOC), voltage_3D)
        return self.response_volume

    def get_response_surface(self, data=None): # need response surface for charge and discharge X2
        power_2D, voltage_2D, amp_rows, SOC_cols = self.process_data2(data, load_existing_data)
        response_surface_dis = RegularGridInterpolator((amp_rows, SOC_cols), voltage_2D)
        response_surface_charge = None
        self.response_surface = (response_surface_dis, response_surface_charge)
        return self.response_surface

    @staticmethod
    def process_data(data=None):
        """Data processing for response volume. (If temperature is to be included)"""
        i, j = 0, 0
        # SOC = np.linspace(0, 1, 25)
        # amps = np.linspace(0, 8, 5)
        # temp = np.linspace(0, 50, 6)
        var_x, var_y, var_z = np.meshgrid(temp, amps, SOC)
        value_matrix3D = np.zeros((len(temp), len(amps), len(SOC)))  # pre-allocate 3-D matrix for speed
        power_matrix_3D = np.zeros((len(temp), len(amps), len(SOC)))

        for key, value in data.items():  # key: Voltage_temp  Value: Columbs moved/SOC
            SOC_values = value[:, 0]/np.max(value[:, 0])
            if len(value) < 30:
                voltage_values = np.flip(np.interp(SOC, SOC_values, value[:, 1]))
                value = np.hstack([SOC_values, voltage_values])
            else:
                SOC_values = np.flip(value[:, 0] / np.max(value[:, 0]))
                value[:, 0] = SOC_values
                voltage_values = np.flip(value[:, 1])
            if i > 5:
                i = 0  # this is for current
                j += 1
            value_matrix3D[i, j, :] = voltage_values
            power_matrix_3D[i, j, :] = voltage_values * amps[i]
            i += 1

        return power_matrix_3D, value_matrix3D, var_x, var_y, var_z

    @staticmethod
    def process_data2(data=None, load_existing_data=False):
        """Data processing for response surface. for discharge
        Discharge is limited by minimum discharge voltage imposed by battery manufacturer."""
        if load_existing_data:
            power_matrix_2D = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/power_2d_discharge.npy')
            value_matrix2D = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/voltage_2d_discharge.npy')
            currents = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/amp_rows_discharge.npy')
            SOC_eval = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/SOC_cols_discharge.npy')
            return power_matrix_2D, value_matrix2D, currents, SOC_eval

        else:
            i = 0
            C = 1  # placeholder
            currents = [0, 5.0, 10.0, 15.0, 25.0]  # Amps
            SOC_eval = np.linspace(0, 1, 200)
            # var_x, var_y = np.meshgrid(currents, SOC_eval)
            value_matrix2D = np.zeros((len(currents), len(SOC_eval)))
            power_matrix_2D = np.zeros((len(currents), len(SOC_eval)))
            for key, value in data.items():
                print("DATA", data)
                SOC_values = np.hstack((0, np.flip(value[:-1, 0])))  # insert 0 percent SOC
                V_values = np.hstack((2.5, np.flip(value[1:, 1])))  # insert final cutoff voltage
                voltage_values = np.interp(SOC_eval, SOC_values, V_values)
                value_matrix2D[i, :] = voltage_values
                power_matrix_2D[i, :] = voltage_values * currents[i]
                i += 1
            np.save('/home/ec2-user/EV50_cosimulation/BatteryData/power_2d_discharge.npy', power_matrix_2D)
            np.save('/home/ec2-user/EV50_cosimulation/BatteryData/voltage_2d_discharge.npy', value_matrix2D)
            np.save('/home/ec2-user/EV50_cosimulation/BatteryData/amp_rows_discharge.npy', currents)
            np.save('/home/ec2-user/EV50_cosimulation/BatteryData/SOC_cols_discharge.npy', SOC_eval)
            return power_matrix_2D, value_matrix2D, currents, SOC_eval

    @staticmethod
    def process_data3(data, load_existing_data=False):
        """Data processing for response surface for charging.
        Charge is limited by maximum c-rate imposed (only 1C data available) by battery manufacturer."""
        if load_existing_data:
            power_matrix_2D = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/power_2d_charge.npy')
            value_matrix2D = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/voltage_2d_charge.npy')
            currents = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/amp_rows_charge.npy')
            SOC_eval = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/SOC_cols_charge.npy')
            return power_matrix_2D, value_matrix2D, currents, SOC_eval

        else:
            i = 0
            C = 1  # placeholder
            currents = [0, 5.0, 10.0, 15.0, 25.0]  # Amps: OCV, 1C, 2C, 3C, 5C
            SOC_eval = np.linspace(0.99, 0, 200)
            # var_x, var_y = np.meshgrid(currents, SOC_eval)
            value_matrix2D = np.zeros((len(currents), len(SOC_eval)))
            power_matrix_2D = np.zeros((len(currents), len(SOC_eval)))
            for key, value in data.items():
                SOC_values = np.hstack((value[1:, 0], 0))  # insert 0 percent SOC
                V_values = np.hstack((value[:-1, 1], 2.5))  # insert final cutoff voltage
                voltage_values = np.interp(SOC_eval, SOC_values, V_values)
                value_matrix2D[i, :] = voltage_values
                power_matrix_2D[i, :] = voltage_values * currents[i]
                i += 1
            np.save('/home/ec2-user/EV50_cosimulation/BatteryData/power_2d_charge.npy', power_matrix_2D)
            np.save('/home/ec2-user/EV50_cosimulation/BatteryData/voltage_2d_charge.npy', value_matrix2D)
            np.save('/home/ec2-user/EV50_cosimulation/BatteryData/amp_rows_charge.npy', currents)
            np.save('/home/ec2-user/EV50_cosimulation/BatteryData/SOC_cols_charge.npy', SOC_eval)
            return power_matrix_2D, value_matrix2D, currents, SOC_eval

BM = BatteryMaps()
BM = BM.get_response_surface()[0]
print(BM([0.1, 0.2]))
print(BM([0, 0.1999999]))  # testing to ensure accuracy for response surface
print(BM([8, 0.198]))
print(BM([4, 0.1999999]))
print(BM([0.00001, 0.1999999]))
print(BM([0.0011, 0.1999999]))
print(BM([0.00011, 0.1999999]))
