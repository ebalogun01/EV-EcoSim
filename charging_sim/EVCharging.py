from chargingStation import ChargingStation
import json
import os
import numpy as np
import random
from utils import add_power_profile_to_object
from plots import plot_results
from battery import Battery
from batteryAgingSim import BatterySim
import controller as control    # FILE WITH CONTROL MODULE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from electricityPrices import PriceLoader

minute_in_day = 1440

class ChargingSim:
    def __init__(self, num_charging_sites, battery_config=None, resolution=15):
        """Design charging sim as orchestrator for battery setup"""
        # TODO: fix these literal paths below
        data2018 = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2018.csv')
        charge_data = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/CP_historical_data_2015_2017.csv')
        test_data = data2018[:-1, ] / 10  # removing bad data
        self.charge_data = charge_data
        self.battery_config = None
        self.charging_config = None
        self.controller_config = None
        self.prices_config = None
        self.price_loader = None
        self.battery_specs_per_loc = None  # Could be used later to specify varying batteries for various nodes
        self.day_year_count = -1
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
        self._nodes = []

        scaler = MinMaxScaler()
        onestep_data = np.reshape(charge_data, (charge_data.size, 1))
        scaler.fit(onestep_data)
        self.scaled_test_data_onestep = [scaler.transform(np.reshape(test_data, (test_data.size, 1))), scaler]

    def load_config(self):
        # use master config for loading other configs also change all these paths from literal
        configs_path = "/home/ec2-user/EV50_cosimulation/charging_sim/configs"
        current_working_dir = os.getcwd()
        os.chdir(configs_path)
        for root, dirs, files, in os.walk(configs_path):
            for file in files:
                attribute = file.split(".")[0] + "_config"
                with open(file, "r") as f:
                    config = json.load(f)
                    setattr(self, attribute, config)
        os.chdir(current_working_dir)    # return back to current working directory

    def create_battery_object(self, Q_initial, loc):
        #  this stores all battery objects in the network
        self.load_config()  # loads existing config files
        buffer_battery = Battery("Tesla Model 3", Q_initial, config=self.battery_config)
        buffer_battery.id, buffer_battery.node = loc, loc
        self.battery_objects.append(buffer_battery)
        buffer_battery.num_cells = buffer_battery.battery_setup(voltage=375, capacity=1500,
                                                                cell_params=(buffer_battery.nominal_voltage, 4.85))
        return buffer_battery

    def create_charging_stations(self, power_nodes_list):
        # add flexibility for multiple units at one charging node?
        # No need, can aggregate them and have a different arrival sampling method
        self.num_charging_sites = min(len(power_nodes_list), self.num_charging_sites)
        loc_list = random.sample(power_nodes_list, self.num_charging_sites)
        # print("There are", len(self.battery_objects), "battery objects initialized")
        for i in range(self.num_charging_sites):
            battery = self.create_battery_object(3.5, loc_list[i])  # change this from float param to generic
            controller = control.MPC(self.controller_config, battery)   # need to change this to load based on the users controller python file?
            assert isinstance(battery, object)  # checks that battery is an obj
            self.charging_config["locator_index"], self.charging_config["location"] = i, loc_list[i]
            charging_station = ChargingStation(battery, self.charging_config, controller)
            self.charging_sites[loc_list[i]] = charging_station
            self.battery_objects.append(battery)    # add to list of battery objects
        self.stations_list = list(self.charging_sites.values())
        self.charging_locs = list(self.charging_sites.keys())

    def initialize_aging_sim(self):
        # TODO: make the number of steps a passed in variable
        self.aging_sim = BatterySim(0, 1)

    def initialize_price_loader(self):
        """Can add option for each charging site to have its own price loader"""
        configs_path = "/home/ec2-user/EV50_cosimulation/charging_sim/configs"
        current_working_dir = os.getcwd()
        self.price_loader = PriceLoader(self.prices_config)
        input_data_res = self.prices_config["resolution"]
        if input_data_res > self.resolution:
            self.price_loader.downscale(input_data_res, self.resolution)
            self.prices_config["resolution"] = self.resolution
            file_path_list = self.prices_config["data_path"].split("_")
            new_data_path = self.prices_config["data_path"].replace(file_path_list[-1], str(self.resolution)+"min.csv")
            self.prices_config["data_path"] = new_data_path
            print(new_data_path)
            with open(self.prices_config["config_path"], 'w') as config_file_path:
                os.chdir(configs_path)
                json.dump(self.prices_config, config_file_path, indent=1)
            os.chdir(current_working_dir)

    def reset_loads(self):
        self.site_net_loads = []

    def get_charging_sites(self):
        return self.charging_locs

    def get_charger_obj_by_loc(self, loc):
        return self.charging_sites[loc]

    def setup(self, power_nodes_list):
        self.create_charging_stations(power_nodes_list)
        self.initialize_price_loader()
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

    def initialize_controllers(self):
        """assign charging controller to each EVSE"""
        # for charging_station in self.stations_list:
        #     charging_station.controller = MPC(self.controller_config, 0, )
        pass

    def step(self, num_steps):
        # NEED TO ADD STEPPING THROUGH DAYS # should this be done in some controller master sim?
        # Full day prediction is not changing but the price is changing!! issues
        """Step forward once. Run MPC controller and take one time-step action.."""
        self.reset_loads()  # reset the loads from old time-step
        elec_price_vec = self.price_loader.get_prices(0, self.num_steps)    # need to freeze daily prices
        for charging_station in self.stations_list:
            if self.time % 96 == 0:
                print("reset")
                charging_station.controller.reset_load()
                elec_price_vec = self.price_loader.get_prices(self.time, self.num_steps)
                self.time = 0  # reset time
                self.day_year_count = 0    # bug..finish this tonight!!

            buffer_battery = charging_station.storage
            run_count = 0
            controls = []
            self.control_start_index = num_steps * self.day_year_count
            stop = self.day_year_count + 14  # get today's load from test data; move to load generator
            self.control_shift = 0
            todays_load = self.test_data[stop]
            # print(todays_load.shape, "load shape")
            # print("today's load", todays_load)
            assert todays_load.size == 96
            todays_load.shape = (todays_load.size, 1)
            loads = [] # TAKE THIS AWAY FROM THE CONTROLLER!
            for i in range(num_steps):
                # print(todays_load[i], "loadcheck")
                # I need to abstract the entire controller away!!!
                control_action, predicted_load = charging_station.controller.compute_control(self.control_start_index,
                                                         self.control_shift, stop, buffer_battery.size, elec_price_vec)
                loads.append(predicted_load[i, 0])
                charging_station.controller.load = np.append(charging_station.controller.load, todays_load[i])
                # update load with the true load, not prediction,
                # to update MPC last observed load
                controls.append(control_action[0] + control_action[1])
                print("CONTROL: ", control_action[0], control_action[1])
                net_load = todays_load[self.time, 0] + (control_action[0] + control_action[1])
                print("time", self.time)
                charging_station.update_load(todays_load[self.time, 0])  # set current load for charging station
                self.update_site_loads(net_load)  # Global Load Monitor for all the loads for this time-step
                # print("site net loads shape is: ", len(self.site_net_loads))


            self.control_shift = 0
            # print("MSE is ", (np.average(charging_station.controller.full_day_prediction - todays_load) ** 2) ** 0.5,
            #       "for day ", stop)

            # update season based on run_count (assumes start_time was set to 0)
            run_count += 1
            # print("Number of cycles is {}".format(run_count))
            # total_savings = show_results(total_savings, buffer_battery, energy_prices_TOU, todays_load)
            # check whether one year has passed
            if self.day_year_count % 365 == 0:  # it has already run for 365 days (This is wrt sampling Solar Generation)
                self.day_year_count = 0
                buffer_battery.start = 0

            buffer_battery.update_capacity()  # updates the capacity to include linear aging for previous run
            self.aging_sim.run(buffer_battery)  # run battery response to actions
            buffer_battery.initial_SOC = buffer_battery.SOC.value[1,0] # this is for the entire battery
            buffer_battery.track_SOC(buffer_battery.SOC.value[1,0])
            # print("Current SOC is: ", buffer_battery.initial_SOC)
            # start_time = buffer_battery.start
            EV_power_profile = todays_load[0:num_steps + 1, ] + buffer_battery.power_charge.value[0:num_steps + 1, ] - \
                               buffer_battery.power_discharge.value[0:num_steps + 1, ]
            add_power_profile_to_object(buffer_battery, self.day_year_count, EV_power_profile)  # update power profile
            print("SOH is: ", buffer_battery.SOH)
            buffer_battery.start += 1
        self.time += 1
        # stop += 1  # shifts to the next day
        plt.figure()
        plt.plot(buffer_battery.true_aging)
        plt.plot(np.array(buffer_battery.linear_aging) / 0.2)
        plt.title("true aging and Linear aging")
        plt.legend(["True aging", "Linear aging"])
        plt.savefig("aging_plot.png")
        plt.close()
        return self.site_net_loads

    def load_results_summary(self):
        # TODO: selecting option for desired statistics
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        charging_sites_keys = self.charging_sites.keys()
        option1 = "loads"
        option2 = "storage"
        charging_stations = self.charging_sites.values()
        for charging_station in charging_stations:
            charging_station.visualize(option=option1)
            charging_station.visualize(option=option2)
        for battery in self.battery_objects:
            battery.visualize("SOC")
            # power_profiles = charging_station.storage.get_power_profile(months)
            # for key, values in power_profiles:
            #     plt.plot(values)
            #     plt.xlabel("Timestep")
            #     plt.ylabel("Power (kW)")
            #     plt.savefig("Charging_site_load_profile_{}".format(charging_station.id))
            #     plt.close()

