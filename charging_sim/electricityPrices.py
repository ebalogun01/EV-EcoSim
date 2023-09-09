"""
This module contains the class that loads the electricity price data and structure used for sampling prices during
simulation.

Based on the prices.json config file, this module will load the desired price TOU rate file that will be used in
optimization problem. The default is the PGE_BEV2_S rate file, which is valid for California, however users can load
their own TOU rate file. The prices are loaded into a numpy array and can be sampled from during simulation.
The prices are sampled based on the month of the year and the hour of the day.
"""

import numpy as np
import pandas as pd

# Old (legacy) reference below:
# https://www.pge.com/tariffs/assets/pdf/tariffbook/ELEC_SCHEDS_EV.pdf
# load_profile is a 24x1 array with kWh consumed in each our of the day, starting at 0:00
# Rates in $/kWh based on "Residential TOU Service for Plug-In EV2"
# Rates in $/kWh based on "Commercial TOU Service for Plug-In EV2"
# Peak (weekday) = 4 to 9 PM 
# Partial-peak (weekday) = 3 to 4 PM, 9 to 12 AM
# Off-peak: all other times


class PriceLoader:
    """This class pre-loads prices and is used to sample prices that are used for optimization of EVSE profits/costs
    during charging simulation.

    :type config: dict
    :param config: Configuration dictionary for the price loader.
    :param path_prefix: This string path prefix is obtained first based on your repository location to set the
    correct path for obtaining the data.
    """

    def __init__(self, config, path_prefix=None):
        """
        Initializes the PriceLoader class.

        :type config: dict
        :param config: Configuration dictionary for the price loader.
        :param path_prefix: This string path prefix is obtained first based on your repository location to set the
        correct path for obtaining the data.
        """
        self.path_prefix = path_prefix
        self.config = config
        self.data = pd.read_csv(path_prefix + self.config["data_path"])
        self.data_np = self.data.to_numpy()
        self.month_start_idx = {1: 0, 2: 31, 3: 59, 4: 90, 5: 120, 6: 151, 7: 181, 8: 243, 9: 273, 10: 304, 11: 334,
                                12: 365}
        self.month = -100  # Default value.

    def get_prices(self, start_idx, num_steps, month=7):
        """
        Returns time-of-use (TOU) rate prices from data. This assumes TOU rates do not change day-to-day.

        :param int start_idx: Starting index from which to price vector will start.
        :param int num_steps: Cardinality of the price vector being returned.
        :param int month: Month for which the price vector will be obtained (for example, 1 - Jan, 12 - December).
        :return ndarray price_vector: The TOU price vector, which is a numpy array.
        """
        price_vector = self.data_np[start_idx:start_idx + num_steps]
        price_vector = price_vector.reshape(-1, 1)
        return price_vector

    def set_month_data(self, month):
        """
        Sets the month for which the prices will be obtained.

        :param month: Month to set the data to.
        :return: None.
        """
        if self.month != month:
            self.data_np = self.data.to_numpy()[self.month_start_idx[month] * 96:self.month_start_idx[month + 1] * 96]

    def downscale(self, input_res, output_res):
        """
        Downscales the price data into a finer resolution, similar to the downscaling method in Pandas.
        Typically only used once.

        :param input_res: Resolution of the input data.
        :param output_res: Resolution of the output data.
        :return: None. Saves output data to a csv file.
        """
        input_data_shape = len(self.data_np[:, 0])
        num_repetitions = int(input_res / output_res)
        assert num_repetitions == 4  # JUST AN INITIAL CHECK, REMOVE LATER
        temp_data = np.zeros(input_data_shape * num_repetitions)
        start_idx = 0
        for datapoint in self.data_np:
            # print(datapoint)
            temp_data[start_idx:start_idx + num_repetitions] = datapoint
            start_idx += 4
        self.data = pd.DataFrame(data=temp_data)
        self.data_np = temp_data
        # IMPORTANT: Change the paths below to save new data.
        np.savetxt(self.path_prefix + "/elec_rates/PGE_BEV2_S_annual_TOU_rate_{}min.csv".format(output_res), temp_data)


def main():
    """This is only run to generate new downscaled data or for testing."""
    import os
    import json
    path_prefix = os.getcwd()
    path_prefix = path_prefix[0:path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
    path_prefix = path_prefix.replace('\\', '/')
    with open(path_prefix + '/charging_sim/configs/prices.json', "r") as f:
        config = json.load(f)
    loader = PriceLoader(config, path_prefix=path_prefix)
    desired_res = 15  # units are in minutes
    loader.downscale(config['resolution'], desired_res)


if __name__ == "__main__":
    main()
