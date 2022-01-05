from chargingStation import ChargingStation
import json
import numpy as np
import random
from config import energy_prices_TOU, add_power_profile_to_object, show_results
from plots import plot_results
from battery import Battery
from batteryAgingSim import BatterySim
from controller import MPC
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

minute_in_day = 1440

class ChargingSim:
    def __init__(self, num_charging_sites, battery_config=None, resolution=15):
        """Design charging sim as orchestrator for battery setup"""

        data2015 = np.genfromtxt(
            '/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2015.csv')  # Need to update these dirs.
        data2016 = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2016.csv') # MOVE TO TO CONFIG
        data2017 = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2017.csv')
        data2018 = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2018.csv')
        charge_data = np.vstack([data2015[7:, ], data2016, data2017]) / 10
        test_data = data2018[:-1, ] / 10  # removing bad data
        self.charge_data = charge_data
        self.battery_config = None
        self.charging_config = None
        self.battery_specs_per_loc = None  # Could be used later to specify varying batteries for various nodes
        self.day_year_count = 0
        self.stop = 0
        self.steps = 0
        self.control_start_index = 0
        self.control_shift = 0
        self.time = 0
        self.test_data = test_data
        self.num_charging_sites = num_charging_sites
        self.charging_locs = None
        self.charging_sites = {}
        self.stations_list = []
        self.battery_objects = []
        self.site_net_loads = []
        self.resolution = resolution
        self.num_steps = int(minute_in_day / resolution)
        self.aging_sim = None  # This gets updated later
        y = self.test_data
        self.std_y = np.std(y, 0)
        self.std_y[self.std_y == 0] = 1
        self.mean_y = np.mean(y, 0)
        self.scaled_test_data = (y - self.mean_y) / self.std_y  # network output
        self._nodes = []

        scaler = MinMaxScaler()
        onestep_data = np.reshape(charge_data, (charge_data.size, 1))
        scaler.fit(onestep_data)
        self.scaled_test_data_onestep = [scaler.transform(np.reshape(test_data, (test_data.size, 1))), scaler]

    def load_config(self):
        with open("/home/ec2-user/EV50_cosimulation/charging_sim/battery_config.json", "r") as f:
            battery_config = json.load(f)
        with open("/home/ec2-user/EV50_cosimulation/charging_sim/charging_config.json", "r") as f:
            charging_config = json.load(f)
        self.battery_config = battery_config
        self.charging_config = charging_config

    def create_battery_object(self, Q_initial, loc):
        #  this stores all battery objects in the network
        self.load_config()  # loads existing config files
        buffer_battery = Battery("Tesla Model 3", Q_initial, config=self.battery_config)
        self.battery_objects.append(buffer_battery)
        buffer_battery.id = loc
        buffer_battery.num_cells = buffer_battery.battery_setup(voltage=375, capacity=8000,
                                                                cell_params=(buffer_battery.nominal_voltage, 4.85))
        return buffer_battery

    def create_charging_stations(self, power_nodes_list):
        # add flexibility for multiple units at one charging node?
        # No need, can aggregate them and have a different arrival sampling method
        self.num_charging_sites = min(len(power_nodes_list), self.num_charging_sites)
        loc_list = random.sample(power_nodes_list, self.num_charging_sites)
        # print("There are", len(self.battery_objects), "battery objects initialized")
        power_cap = 100  # kW This should eventually be optional as well
        for i in range(self.num_charging_sites):
            battery = self.create_battery_object(3.5, loc_list[i])
            assert isinstance(battery, object)  # checks that battery is an obj
            self.charging_config["locator_index"] = i
            charging_station = ChargingStation(battery, loc_list[i], power_cap, self.charging_config)
            self.charging_sites[loc_list[i]] = charging_station
            self.battery_objects.append(battery)    # add to list of battery objects
        self.stations_list = list(self.charging_sites.values())
        self.charging_locs = list(self.charging_sites.keys())

    def initialize_aging_sim(self):
        # TODO: make the number of steps a passed in variable
        self.aging_sim = BatterySim(0, self.num_steps)

    def reset_loads(self):
        self.site_net_loads = []

    def get_charging_sites(self):
        return self.charging_locs

    def get_charger_obj_by_loc(self, loc):
        return self.charging_sites[loc]

    def setup(self, power_nodes_list):
        self.create_charging_stations(power_nodes_list)
        self.initialize_aging_sim()  # Battery aging
        
    def update_site_loads(self, load):
        self.site_net_loads.append(load)

    def update_steps(self, steps):
        self.steps += steps
        if self.steps == minute_in_day / self.resolution:
            self.day_year_count += 1

    @staticmethod
    def get_action(self):
        """returns only the control current"""
        return 0

    def step(self, num_steps):
        # NEED TO ADD STEPPING THROUGH DAYS # should this be done in some controller master sim?
        """Step forward once. Run MPC controller and take one time-step action.."""
        self.reset_loads()  # reset the loads from old time-step
        # each charging station is using an instantiation of an MPC controller. I think we should add controller object
        for charging_station in self.stations_list:
            buffer_battery = charging_station.storage
            charging_station.controller = MPC(self.charge_data, self.scaled_test_data,
                          self.scaled_test_data_onestep, buffer_battery, self.std_y, self.mean_y)   # inefficient
            run_count = 0
            controls = []
            self.control_start_index = num_steps * self.day_year_count
            stop = self.day_year_count + 14  # get today's load from test data
            self.control_shift = 0
            todays_load = self.test_data[stop]
            assert todays_load.size == 96
            todays_load.shape = (todays_load.size, 1)
            loads = []
            for i in range(num_steps):
                control_action, predicted_load = charging_station.controller.compute_control(self.control_start_index,
                                                                        self.control_shift, stop, buffer_battery.size)
                loads.append(predicted_load[i, 0])
                charging_station.controller.load = np.append(charging_station.controller.load, todays_load[i])
                # update load with the true load, not prediction,
                # to update MPC last observed load
                controls.append(control_action[0] + control_action[1])
                print("CONTROL: ", control_action[0], control_action[1])
                net_load = todays_load[self.time, 0] + (control_action[0] + control_action[1])
                charging_station.update_load(net_load)  # set current load for charging station
                self.control_start_index += 1
                self.control_shift += 1
                self.update_site_loads(net_load)  # Global Load Monitor for all the loads for this time-step
                # print("site net loads shape is: ", len(self.site_net_loads))

            stop += 1  # shifts to the next day
            self.control_shift = 0
            print("MSE is ", (np.average(charging_station.controller.full_day_prediction - todays_load) ** 2) ** 0.5,
                  "for day ", stop)
            if self.control_shift == 96:
                print("reset")
                charging_station.controller.reset_load()
                self.stop += 1

            # update season based on run_count (assumes start_time was set to 0)
            run_count += 1
            print("Number of cycles is {}".format(run_count))
            # total_savings = show_results(total_savings, buffer_battery, energy_prices_TOU, todays_load)
            # check whether one year has passed
            if self.day_year_count % 365 == 0:  # it has already run for 365 days (This is wrt sampling Solar Generation)
                self.day_year_count = 0
                buffer_battery.start = 0

            buffer_battery.update_capacity()  # updates the capacity to include linear aging for previous run
            self.aging_sim.run(buffer_battery)  # run battery response to actions
            buffer_battery.Q_initial = buffer_battery.Q.value[1][0]  # this is for the entire battery
            start_time = buffer_battery.start
            EV_power_profile = todays_load[0:num_steps + 1, ] + buffer_battery.power_charge.value[0:num_steps + 1, ] - \
                               buffer_battery.power_discharge.value[0:num_steps + 1, ]
            add_power_profile_to_object(buffer_battery, self.day_year_count, EV_power_profile)  # update power profile
            print("SOH is: ", buffer_battery.SOH)
            buffer_battery.start += 1
        self.time += 1
        plt.figure()
        plt.plot(buffer_battery.true_aging)
        plt.plot(np.array(buffer_battery.linear_aging) / 0.2)
        plt.title("true aging and Linear aging")
        plt.legend(["True aging", "Linear aging"])
        plt.savefig("aging_plot.png")
        plt.close()
        return self.site_net_loads
