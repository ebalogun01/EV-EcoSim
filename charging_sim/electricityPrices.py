"""Contains pricing structure and electricity data. Should contain attribute that spits out vector of prices depending
on whatever pricing scheme is being used..."""
import numpy as np
from dataclasses import dataclass
import pandas as pd


# load_profile is a 24x1 array with kWh consumed in each our of the day, starting at 0:00
# Rates in $/kWh based on "Residential TOU Service for Plug-In EV2"
# Peak (weekday) = 4 to 9 PM 
# Partial-peak (weekday) = 3 to 4 PM, 9 to 12 AM
# Off-peak: all other times
@dataclass
class PriceLoader:
    """module for laoding prices, Undecided if this should. UPDATE HARD-CODED PATHS"""
    def __init__(self, config, path_prefix=None):
        self.path_prefix = path_prefix
        self.config = config
        self.data = pd.read_csv(path_prefix+self.config["data_path"])
        self.data_np = self.data.to_numpy()

    def get_prices(self, start_idx, num_steps, desired_shape=(96, 1)):
        price_vector = self.data_np[start_idx:start_idx+num_steps]
        price_vector = np.reshape(price_vector, desired_shape)
        return price_vector

    def downscale(self, input_res, output_res):
        input_data_shape = len(self.data_np[:, 0])
        num_repetitions = int(input_res / output_res)
        assert num_repetitions == 4     # JUST AN INITIAL CHECK, REMOVE LATER
        temp_data = np.zeros(input_data_shape * num_repetitions)
        start_idx = 0
        for datapoint in self.data_np:
            print(datapoint)
            temp_data[start_idx:start_idx + num_repetitions] = datapoint
            start_idx += 4
        self.data = pd.DataFrame(data=temp_data)
        self.data_np = temp_data
        # change the paths below very soon
        np.savetxt(self.path_prefix+"/charging_sim/annual_TOU_rate_{}min.csv".format(output_res), temp_data)


def main():
    """Run this mainly to generate new downscaled data or for testing"""
    import os
    import json
    path_prefix = os.getcwd()
    path_prefix = path_prefix[0:path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
    path_prefix.replace('\\', '/')
    with open(path_prefix+'charging_sim/prices.json', "r") as f:
        config = json.load(f)
    loader = PriceLoader(config, path_prefix=path_prefix)
    desired_res = 1     # units are in minutes
    loader.downscale(config['resolution'], desired_res)
#
# TOU_rate_summer = np.array(
#         [0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611,
#          0.16611, 0.16611, 0.16611, 0.16611, 0.36812, 0.47861, 0.47861, 0.47861, 0.47861, 0.47861,
#          0.36812, 0.36812, 0.36812])
#
# TOU_rate_winter = np.array(
#     [0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611, 0.16611,
#      0.16611, 0.16611, 0.16611, 0.16611, 0.33480, 0.35150, 0.35150, 0.35150, 0.35150, 0.35150,
#      0.33480, 0.33480, 0.33480])

# # summmer - November - March, # Winter - April - October
# Jan = np.tile(TOU_rate_winter, 31)
# Feb = np.tile(TOU_rate_winter, 28)
# March = np.tile(TOU_rate_winter, 31)
# Apr = np.tile(TOU_rate_winter, 30)
# May = np.tile(TOU_rate_summer, 31)
# June = np.tile(TOU_rate_summer, 30)
# July_Aug = np.tile(TOU_rate_summer, 31*2)
# Sep = np.tile(TOU_rate_summer, 30)
# Oct = np.tile(TOU_rate_summer, 31)
# Nov = np.tile(TOU_rate_winter, 30)
# Dec = np.tile(TOU_rate_summer, 31)
# #
# annual_price_vector = np.concatenate((Jan, Feb, March, Apr, May, June, July_Aug, Sep, Oct, Nov, Dec))
# # np.savetxt("annual_TOU_rate.csv", annual_price_vector)
# # print(annual_price_vector.shape, 24*365)