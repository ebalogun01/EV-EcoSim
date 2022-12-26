import numpy as np
import pandas as pd
import cvxpy as cp

class Solar:
    """This simulates the ground-truth Solar conditions in location"""
    def __init__(self, config, path_prefix=None, controller=None):
        self.path_prefix = path_prefix + '/'
        self.config = config
        cols = ['Month', 'Day', 'Hour', 'GHI', 'Temperature']
        self.cols = cols
        self.controller = controller
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
        self.area = self.config["area"]
        self.rating = self.config['rating'] * 1000  # convert from MW to kW
        self.data = None
        self.data_np = None
        self.battery_power = cp.Variable((self.num_steps, 1))  # solar power to battery
        self.ev_power = cp.Variable((self.num_steps, 1))  # solar power to ev
        self.grid_power = cp.Variable((self.num_steps, 1))  # TODO: consider including this in future work
        self.power = None
        self.constraints = []
        self.month = 1
        self.id = None
        self.node = None

    def get_power(self, start_idx, num_steps, desired_shape=(96, 1), month=7):
        if not self.month == month:
            # GHI = Global Horizontal Irradiance
            self.data_np = self.solar_df[self.solar_df["Month"] == month]['GHI'].to_numpy()     # for one month
            self.month = month  # set month to current desired month
        self.power = self.data_np[start_idx:start_idx+num_steps]
        self.power = np.minimum(self.rating, np.reshape(self.power, desired_shape) * self.efficiency)
        print("Solar Max Power is: {}kW".format(self.power.max()))
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
            # print(datapoint)
            temp_data[start_idx:start_idx + num_repetitions] = datapoint
            start_idx += num_repetitions
        self.data = pd.DataFrame(data=temp_data, columns=self.cols)
        self.data_np = temp_data
        path_suffix = self.config['data_path'].split('-')[0]
        self.data.to_csv(self.path_prefix+path_suffix+"-{}min.csv".format(output_res))

    def get_solar_output(self):
        return self.data_np * self.efficiency * self.area / 1000    # this is in kW

    def get_constraints(self):
        if not self.constraints:
            self.constraints = [self.power == self.battery_power + self.ev_power + self.grid_power,
                                self.battery_power >= 0,
                                self.ev_power >= 0,
                                self.grid_power >= 0]
        return self.constraints

    def update_history(self):
        #TODO: add solar history update
        return NotImplementedError

def main():
    """THis is mainly to testing or generating new data purposes"""
    import os
    import json
    # load input data
    path_prefix = os.getcwd()
    path_prefix = path_prefix[0:path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation'
    path_prefix = path_prefix.replace('\\', '/')
    with open(path_prefix+'/charging_sim/configs/solar.json', "r") as f:
        config = json.load(f)
    solar = Solar(config, path_prefix=path_prefix)
    desired_res = 15
    solar.downscale(30, desired_res)


if __name__ == "__main__":
    main()
