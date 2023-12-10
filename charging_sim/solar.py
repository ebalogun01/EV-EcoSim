"""
This module contains the Solar class. The solar class is used to simulate the solar power generation at a given site
by sampling Global Horizontal Irradiance (GHI) data from the dataset of the desired location.
"""

import numpy as np
import pandas as pd
import cvxpy as cp


class Solar:
    """
    This class is used to simulate the solar power generation at a given site by sampling Global Horizontal Irradiance
    and estimating the solar generation, given the solar nameplate capacity that can be modified it its configuration
    file `solar.json`. It also contains the optimization variables for the solar system.

    The solar power generation is estimated using the following equation:

    .. math::
        P_{solar} = \min(P_{rated}, \eta *  A * GHI).

    Where :math:`P_{solar}` is the solar power generation, :math:`\eta` is the efficiency of the solar system,
    :math:`P_{rated}` is the solar nameplate capacity, :math:`A` is the area of the solar panels, and :math:`GHI` is
    the Global Horizontal Irradiance.

    :param config: Solar configuration dictionary.
    :param path_prefix: This string path prefix is obtained first based on your repository location to set the right path.
    :param controller: Controller object for making decisions on flow of power from energy devices.
    :param num_steps: Number of steps in the simulation.

    """

    def __init__(self, config, path_prefix=None, controller=None, num_steps=None):
        self.path_prefix = path_prefix + '/'
        self.config = config
        cols = ['Month', 'Day', 'Hour', 'GHI', 'Temperature']
        self.cols = cols
        self.controller = controller
        if num_steps:
            self.num_steps = num_steps
        else:
            self.num_steps = self.config["num steps"]
        self.solar_df = pd.read_csv(self.path_prefix+self.config["data_path"])[cols]
        self.solar_vec = self.solar_df.to_numpy()
        self.location = self.config["location"]
        self.start_year = self.config["start_year"]
        self.start_month = self.config["start_month"]
        self.start_day = self.config["start_day"]
        self.efficiency = self.config["efficiency"]
        self.resolution = self.config["resolution"]
        self.input_data_res = self.config["input_res"]
        self.sample_start_idx = self.config["sample_start_idx"]
        self.sample_end_idx = self.config["sample_end_idx"]
        self.rating = self.config['rating'] * 1000  # convert from MW to kW
        self.area = self.rating * 1000 / 150    # approximate area using 150W/m2
        self.data = None
        # self.data_np = None
        self.data_np = self.solar_df[self.solar_df["Month"] == self.start_month]['GHI'].to_numpy()
        self.battery_power = cp.Variable((self.num_steps, 1), nonneg=True)  # solar power to battery
        self.ev_power = cp.Variable((self.num_steps, 1), nonneg=True)  # solar power to ev
        self.grid_power = cp.Variable((self.num_steps, 1), nonneg=True)  # TODO: consider including this in future work
        self.power = None
        self.constraints = []
        self.month = self.start_month   # initializing for the start of the simulation
        self.id = None
        self.node = None

    def get_power(self, start_idx, num_steps, desired_shape=(96, 1), month=None):
        if month is not None and self.month != month:
            # GHI = Global Horizontal Irradiance
            print("setting month for solar power...")
            self.data_np = self.solar_df[self.solar_df["Month"] == month]['GHI'].to_numpy()     # for one month
            self.month = month  # set month to current desired month
        self.power = self.data_np[start_idx:start_idx+num_steps] / 1000     # convert to kW
        self.power = np.minimum(self.rating, np.reshape(self.power, desired_shape) * self.efficiency * self.area)
        # this ignores area for now. Can look at potential land-use/space use in future work
        return self.power

    def modify_res(self, new_res):
        pass

    def downscale(self, input_res, output_res):
        """This is used only to initially downscale data to desired resolution"""
        num_repetitions = int(input_res / output_res)
        assert num_repetitions == 2  # JUST AN INITIAL CHECK, REMOVE LATER
        temp_data = np.zeros((self.solar_vec.shape[0]*num_repetitions, self.solar_vec.shape[1]))
        start_idx = 0
        for datapoint in self.solar_vec:
            temp_data[start_idx:start_idx + num_repetitions] = datapoint
            start_idx += num_repetitions
        self.data = pd.DataFrame(data=temp_data, columns=self.cols)
        self.data_np = temp_data
        path_suffix = self.config['data_path'].split('-')[0]
        self.data.to_csv(self.path_prefix+path_suffix + f"-{output_res}min.csv")

    def get_solar_output(self):
        return self.data_np * self.efficiency * self.area / 1000    # this is in kW

    def get_constraints(self):
        assert self.power is not None
        self.constraints = [self.battery_power + self.ev_power + self.grid_power == self.power]
        return self.constraints

    def update_history(self):
        # CODE IS NOT UPDATING ALL THE ACTIONS SO THERE IS A BUG THERE
        #TODO: add solar history update
        return NotImplementedError


def main():
    """THis is mainly to testing or generating new data purposes"""
    import os
    import json
    # load input data
    path_prefix = os.getcwd()
    path_prefix = path_prefix[:path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
    path_prefix = path_prefix.replace('\\', '/')
    with open(path_prefix+'/charging_sim/configs/solar.json', "r") as f:
        config = json.load(f)
    solar = Solar(config, path_prefix=path_prefix)
    desired_res = 15
    #   input res, desired res
    solar.downscale(30, desired_res)


if __name__ == "__main__":
    main()
