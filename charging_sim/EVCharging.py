from chargingStation import ChargingStation
import json
import os
import numpy as np
import random
from batterypack import Battery
from batteryAgingSim import BatterySim
import controller as control  # FILE WITH CONTROL MODULE
import matplotlib.pyplot as plt
from electricityPrices import PriceLoader
from solar import Solar

# plt.style.use('seaborn-darkgrid')  # optional
minute_in_day = 1440

# TODO: COMPLETE SEPERATE CONFIGS FOR EACH CHARGING STATION GIVEN CONFIG (fast charging vs L2) and include relevant load data
# clean code COMPLETELY
# TODO: track the loads as well

class ChargingSim:
    def __init__(self, num_charging_sites, solar=True, resolution=15, path_prefix=None, num_steps=None, month=6):
        """Design charging sim as orchestrator for battery setup"""
        # TODO: 3/14/2023 changed loads to generated loads from Speech
        num_evs = 200
        self.month = month
        if solar:
            self.solar = True  # to be initialized with module later
        data2018 = np.loadtxt(f'{path_prefix}/speechLoadData/speechWeekdayLoad{num_evs}.csv')  # this is only 30 days data
        print('SpeechData loaded...')
        charge_data = np.loadtxt(f'{path_prefix}/speechLoadData/speechWeekdayLoad{num_evs}.csv')
        # charge_data = np.genfromtxt(f'{path_prefix}/CP_ProjectData/power_data_2018.csv')[:-1, ]/4
        # test_data = data2018[:-1, ]  # removing bad data
        self.path_prefix = path_prefix
        self.charge_data = charge_data
        self.solar_config = None
        self.battery_config = None
        self.charging_config = None
        self.prices_config = None
        self.price_loader = None
        self.battery_specs_per_loc = None  # Could be used later to specify varying batteries for various nodes
        self.day_year_count = 0
        self.stop = 0
        self.steps = 0
        self.control_start_index = 0
        self.control_shift = 0
        self.time = 0
        self.test_data = charge_data.flatten()
        self.num_charging_sites = num_charging_sites
        self.charging_locs = []
        self.charging_sites = {}
        self.stations_list = []
        self.battery_objects = []
        self.storage_locs = []  # usually empty if not centralized mode
        self.site_net_loads = []
        self.resolution = resolution
        if not num_steps:
            self.num_steps = int(minute_in_day / resolution)
        else:
            self.num_steps = num_steps
        self.aging_sim = None  # This gets updated later
        self._nodes = []
        self.scenario = None    # to be updated later

    def load_config(self):
        # use master config for loading other configs also change all these paths from literal
        configs_path = f'{self.path_prefix}/charging_sim/configs'
        current_working_dir = os.getcwd()
        os.chdir(configs_path)
        for root, dirs, files, in os.walk(configs_path):
            for file in files:
                attribute = file.split(".")[0] + "_config"
                with open(file, "r") as f:
                    config = json.load(f)
                    setattr(self, attribute, config)
        os.chdir(current_working_dir)  # return back to current working directory
        self.load_battery_params()  # update the battery params to have model dynamics for all cells loaded already

    def load_battery_params(self):
        """ This loads the battery params directly into the sim, so parameters will be the same for all
        batteries unless otherwise specified. battery_config must be attributed to do this"""
        params_list = [key for key in self.battery_config.keys() if "params_" in key]
        for params_key in params_list:
            self.battery_config[params_key] = np.loadtxt(
                self.path_prefix + self.battery_config[params_key])  # replace path with true value
        # do the OCV maps as well; reverse directionality is important for numpy.interp function
        self.battery_config["OCV_map_voltage"] = np.loadtxt(self.path_prefix + self.battery_config["OCV_map_voltage"])[
                                                 ::-1]
        self.battery_config["OCV_map_SOC"] = np.loadtxt(self.path_prefix + self.battery_config["OCV_map_SOC"])[::-1]

        # this should make those inputs just be the params

    def create_battery_object(self, idx, node_prop, controller=None):
        #  this creates and stores all battery objects in the network
        buffer_battery = Battery(config=self.battery_config, controller=controller)
        buffer_battery.id, buffer_battery.node = idx, node_prop['node']
        buffer_battery.num_cells = buffer_battery.battery_setup()  # toggle between setup and setuo_2 to scale kWh energy capacity using voltage changed this to try scaling voltage instead
        buffer_battery.load_pack_props()    # this is for simulating the entire pack at once
        self.battery_objects += buffer_battery,  # add to list of battery objects
        return buffer_battery

    def create_charging_stations(self, power_nodes):
        # todo: upgrade this in the future to account for variable steps
        # add flexibility for multiple units at one charging node?
        # No need, can aggregate them and have a different arrival sampling method
        loc_list = power_nodes
        # make a list of dicts with varying capacities
        # todo: change the starting point based on existing charging stations
        for i in range(len(loc_list)):
            battery = self.create_battery_object(i, loc_list[i])  # change this from float param to generic
            solar = self.create_solar_object(i, loc_list[i])  # create solar object
            self.controller_config['opt_solver'] = self.scenario['opt_solver']  # set the optimization solver
            controller = control.MPC(self.controller_config, storage=battery,
                                     solar=solar)  # need to change this to load based on the users controller python file?
            self.charging_config['locator_index'], self.charging_config['location'] = i, loc_list[i]['node']
            self.charging_config['L2'] = loc_list[i]['L2']
            self.charging_config['DCFC'] = loc_list[i]['DCFC']
            charging_station = ChargingStation(battery, self.charging_config, controller, solar=solar)
            print(loc_list)
            self.charging_sites[loc_list[i]['node']] = charging_station
        self.stations_list = list(self.charging_sites.values())
        self.charging_locs = list(self.charging_sites.keys())

    def create_charging_stations_oneshot(self, power_nodes):
        # todo: update this for oneshot simulation
        loc_list = power_nodes
        # make a list of dicts with varying capacities
        # todo: change the starting point based on existing charging stations
        for i in range(len(loc_list)):
            battery = self.create_battery_object(i, loc_list[i])  # change this from float param to generic
            solar = self.create_solar_object(i, loc_list[i])  # create solar object
            self.controller_config['opt_solver'] = self.scenario['opt_solver']  # set the optimization solver
            controller = control.Oneshot(self.controller_config, storage=battery,
                                     solar=solar, num_steps=self.num_steps)  # need to change this to load based on the users controller python file?
            self.charging_config['locator_index'], self.charging_config['location'] = i, loc_list[i]['node']
            self.charging_config['L2'] = loc_list[i]['L2']
            self.charging_config['DCFC'] = loc_list[i]['DCFC']
            charging_station = ChargingStation(battery, self.charging_config, controller, solar=solar)
            print(loc_list)
            self.charging_sites[loc_list[i]['node']] = charging_station
        self.stations_list = list(self.charging_sites.values())
        self.charging_locs = list(self.charging_sites.keys())

    def initialize_aging_sim(self):
        # TODO: make the number of steps a passed in variable
        num_steps = 1
        self.aging_sim = BatterySim(0, num_steps)

    def initialize_price_loader(self, month):
        """Loads the price loading module and sets the month to be simulated for memory-efficient sampling"""
        configs_path = f'{self.path_prefix}/charging_sim/configs'
        current_working_dir = os.getcwd()
        self.price_loader = PriceLoader(self.prices_config, path_prefix=self.path_prefix)
        self.price_loader.set_month_data(month)
        input_data_res = self.prices_config["resolution"]
        if input_data_res > self.resolution:
            self.price_loader.downscale(input_data_res, self.resolution)
            self.prices_config["resolution"] = self.resolution
            file_path_list = self.prices_config["data_path"].split("_")
            new_data_path = self.prices_config["data_path"].replace(file_path_list[-1],str(self.resolution) + "min.csv")
            self.prices_config["data_path"] = new_data_path
            with open(self.prices_config["config_path"], 'w') as config_file_path:
                os.chdir(configs_path)
                json.dump(self.prices_config, config_file_path, indent=1)
            os.chdir(current_working_dir)

    def initialize_solar_module(self):
        """This initializes the solar module if there is solar"""
        if self.solar:
            self.solar = Solar(self.solar_config, path_prefix=self.path_prefix, num_steps=self.num_steps)
            print("Solar module loaded...")
        else:
            raise IOError('Cannot load solar module because flag is set to False!')

    def create_solar_object(self, idx, loc, controller=None):
        solar = Solar(config=self.solar_config, path_prefix=self.path_prefix,
                      controller=controller, num_steps=self.num_steps)  # remove Q_initial later
        solar.id, solar.node = idx, loc  # using one index to represent both id and location
        return solar

    def reset_loads(self):
        self.site_net_loads = []

    def get_charging_sites(self):
        return self.charging_locs

    def get_charger_obj_by_loc(self, loc):
        return self.charging_sites[loc]

    def get_storage_sites(self):
        return self.storage_sites

    def setup(self, power_nodes_list, scenario=None):
        # changing power nodes list to dict to distinguish L2 for DCFC
        """This is used to set up charging station locations and simulations"""
        self.load_config()  # FIRST LOAD THE CONFIG ATTRIBUTES
        self.update_scenario(scenario)  # scenarios for study
        self.scenario = scenario
        if self.scenario:
            if 'oneshot' in list(scenario.keys()):
                print("One shot optimization loading...")
                self.create_charging_stations_oneshot(power_nodes_list)     # this allows to load controller the right way
            else:
                self.create_charging_stations(power_nodes_list)  # this should always be first since it loads the config
            self.initialize_price_loader(self.prices_config["month"])
            self.initialize_aging_sim()  # Battery aging
            self.initialize_solar_module()  # this loads solar module (LAST is important for oneshot opt)

    def update_scenario(self, scenario=None):
        if scenario:
            self.prices_config['month'] = scenario['start_month']
            if self.solar_config:
                self.solar_config['start_month'] = scenario['start_month']
            for key in scenario.keys():
                if key != 'index':
                    self.battery_config[key] = scenario[key]
            print('New scenario updated...')

    def update_site_loads(self, load):
        self.site_net_loads += load,

    def update_steps(self, steps):
        self.steps += steps
        if self.steps == minute_in_day / self.resolution:
            self.day_year_count += 1

    @staticmethod
    def get_action(self):
        """returns only the control current"""
        raise NotImplementedError("Function not implemented yet!")

    def initialize_controllers(self):
        """assign charging controller to each EVSE"""
        raise NotImplementedError("Function not implemented yet!")

    def step(self, stepsize):
        """Perfect foresight daily stepping...should I do full-shot run (one optimization per day?)"""
        self.reset_loads()  # reset the loads from old time-step
        elec_price_vec = self.price_loader.get_prices(self.time, self.num_steps)    # time already accounted for
        for charging_station in self.stations_list:  # TODO: how can this be efficiently parallelized ?
            if self.time % 96 == 0:
                charging_station.controller.reset_load()
                self.time = 0  # reset time
            p = charging_station.controller.solar.get_power(self.time, self.num_steps)  # can set month for multi-month sim later
            buffer_battery = charging_station.storage
            self.control_start_index = stepsize * self.day_year_count
            todays_load = self.test_data[self.day_year_count*self.num_steps+self.time
                                         :self.num_steps*(self.day_year_count+1)+self.time] * 1 # this is where time comes in
            todays_load = np.minimum(todays_load, charging_station.capacity)   # only accept capacity of charging station
            todays_load = todays_load.reshape(-1, 1)
            for i in range(stepsize):
                control_action = charging_station.controller.compute_control(todays_load, elec_price_vec)
                charging_station.controller.load += todays_load[i],
                buffer_battery.dynamics(control_action)
                net_load = todays_load[0, 0] + buffer_battery.power - charging_station.solar.power[0, 0]   # moving horizon so always only pick the first one
                charging_station.update_load(net_load, todays_load[0, 0])   # in kW
            # check whether one year has passed (not relevant since we don't run one full year yet)
            if self.day_year_count % 365 == 0:
                self.day_year_count = 0
            buffer_battery.update_capacity()  # update the linear aging vector to include linear aging for previous run
            self.aging_sim.run(buffer_battery)  # simulate battery aging
            charging_station.controller.battery_initial_SOC = charging_station.controller.battery_SOC.value[1, 0]
        self.time += 1
        self.update_steps(stepsize)
        return self.site_net_loads

    def multistep(self):
        elec_price_vec = self.price_loader.get_prices(self.time, self.num_steps, desired_shape=(self.num_steps, 1))  # time already accounted for
        for charging_station in self.stations_list:  # TODO: how can this be efficiently parallelized ?
            p = charging_station.controller.solar.get_power(self.time, self.num_steps, desired_shape=(self.num_steps, 1))  # can set month for multi-month sim later
            buffer_battery = charging_station.storage
            todays_load = self.test_data[self.day_year_count * self.num_steps + self.time
                                         :self.num_steps * (self.day_year_count + 1) + self.time] * 1  # this is where time comes in
            todays_load = np.minimum(todays_load, charging_station.capacity)
            todays_load = todays_load.reshape(-1, 1)
            control_actions = charging_station.controller.compute_control(todays_load, elec_price_vec)
            charging_station.controller.load = todays_load
            battery_powers = []
            for interval in range(self.num_steps):
                charging_station.storage.dynamics(control_actions[interval, 0])
                self.aging_sim.run(buffer_battery)
                battery_powers.append(buffer_battery.power)
            net_load = todays_load + np.array(buffer_battery.power).reshape(-1, 1) - charging_station.solar.power
            charging_station.update_load_oneshot(net_load, todays_load)

    def load_results_summary(self, save_path_prefix, plot=False):
        # TODO: selecting option for desired statistics
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for charging_station in self.stations_list:
            charging_station.save_sim_data(save_path_prefix)
        for battery in self.battery_objects:
            battery.save_sim_data(save_path_prefix)

        # SOME ADDITIONAL PLOTS BELOW
        if plot:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(battery.SOC_track[1:], label='battery SoC')
            ax2.plot(battery.calendar_aging[1:], color='r', ls='--', label='calendar aging')
            ax1.legend()
            ax2.legend()
            ax1.set_xlabel('Time step')
            ax1.set_ylabel('SOC')
            ax2.set_ylabel('Calendar aging')
            plt.savefig(save_path_prefix + "/Calendar_SOC_plot.png")
            plt.close('all')

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(battery.SOC_track[1:], label='battery SOC')
            ax2.plot(battery.cycle_aging[1:], color='r', ls='--', label='cycle aging')
            ax1.set_xlabel('Time step')
            ax1.set_ylabel('SOC')
            ax2.set_ylabel('Cycle aging')
            ax1.legend()
            ax2.legend()
            plt.savefig(save_path_prefix + "/Cycle_SOC_plot.png")
            plt.close('all')

        print("total calendar aging is {}".format(sum(battery.calendar_aging)))
        print("total cycle aging is {}".format(sum(battery.cycle_aging)))
        print("Final capacity (SOH) is {}".format(battery.SOH))
        print("Total current throughput is {}".format(battery.total_amp_thruput))


class ChargingSimCentralized:
    """This module is for centralized storage simulation"""

    def __init__(self, num_charging_sites, num_storage_sites=1, resolution=15, path_prefix=None):
        """Design charging sim as orchestrator for battery setup"""
        # TODO: fix these literal paths below and leave this for future work
        data2018 = np.genfromtxt(path_prefix + '/CP_ProjectData/power_data_2018.csv')
        charge_data = np.genfromtxt(path_prefix + '/CP_ProjectData/CP_historical_data_2015_2017.csv')
        test_data = data2018[:-1, ] / 10  # removing bad data
        self.path_prefix = path_prefix
        self.charge_data = charge_data
        self.battery_config = None
        self.charging_config = None
        self.controller_config = None
        self.prices_config = None
        self.solar_config = None
        self.price_loader = None
        self.battery_specs_per_loc = None  # Could be used later to specify varying batteries for various nodes
        self.central_storage_controllers = []
        self.central_storage_controller = None
        self.day_year_count = -1  # TODO: correct this later
        self.stop = 0
        self.steps = 0
        self.control_start_index = 0
        self.control_shift = 0
        self.time = 0
        self.test_data = test_data
        self.num_charging_sites = num_charging_sites
        self.num_storage_sites = num_storage_sites
        self.charging_locs = []
        self.storage_locs = []
        self.charging_sites = {}
        self.storage_sites = {}
        self.stations_list = []
        self.battery_objects = []
        self.site_net_loads = []
        self.resolution = resolution
        self.num_steps = int(minute_in_day / resolution)
        self.aging_sim = None  # This gets updated later
        self._nodes = []
        onestep_data = np.reshape(charge_data, (charge_data.size, 1))

    def load_config(self):
        configs_path = f'{self.path_prefix}/charging_sim/configs'
        current_working_dir = os.getcwd()
        os.chdir(configs_path)
        for root, dirs, files, in os.walk(configs_path):
            for file in files:
                attribute = file.split(".")[0] + "_config"
                with open(file, "r") as f:
                    config = json.load(f)
                    setattr(self, attribute, config)
        os.chdir(current_working_dir)  # return back to current working directory
        self.load_battery_params()  # update the battery params to have model dynamics for all cells loaded already

    def load_battery_params(self):
        """ Loads the battery params directly into the sim, so parameters will be the same for all
        batteries unless otherwise specified. battery_config must be attributed to do this"""
        params_list = [key for key in self.battery_config.keys() if "params_" in key]
        for params_key in params_list:
            self.battery_config[params_key] = np.loadtxt(
                self.path_prefix + self.battery_config[params_key])  # replace path with true value
        # do the OCV maps as well; reverse directionality is important for numpy.interp function
        self.battery_config["OCV_map_voltage"] = np.loadtxt(self.path_prefix + self.battery_config["OCV_map_voltage"])[
                                                 ::-1]
        self.battery_config["OCV_map_SOC"] = np.loadtxt(self.path_prefix + self.battery_config["OCV_map_SOC"])[::-1]
        # this should make those inputs just be the params

    def create_battery_object(self, idx, loc, controller=None):
        """stores and tracks all battery objects in the network"""
        buffer_battery = Battery(config=self.battery_config, controller=controller)  # remove Q_initial later
        buffer_battery.id, buffer_battery.node = idx, loc  # using one index to represent both id and location
        self.battery_objects += buffer_battery,  # adding battery object to battery list
        buffer_battery.num_cells = buffer_battery.battery_setup()
        return buffer_battery

    def set_central_storage_controller(self, controller):
        self.central_storage_controller = controller

    def update_scenario(self, scenario=None):
        if scenario:
            pass
        pass

    def create_charging_stations(self, power_nodes_list):
        self.load_config()
        num_central_storage = 1  # make this more as an input in the future
        battery_loc_list = random.sample(power_nodes_list,
                                         num_central_storage)  # decide what node in the grid to place the battery

        for i in range(self.num_storage_sites):  # this will only work for one now
            battery = self.create_battery_object(i, battery_loc_list[i])
            battery_controller = control.MPCBatt(self.controller_config, battery)
            self.set_central_storage_controller(battery_controller)  # initialize central storage controller
            self.storage_sites[battery_loc_list[i]] = battery

        if min(len(power_nodes_list), self.num_charging_sites) < self.num_charging_sites:
            print(
                "WARNING: cannot assign more charging nodes than grid nodes...adjusting to the length of power nodes!")
            self.num_charging_sites = min(len(power_nodes_list), self.num_charging_sites)
        # TODO: remove the battery from being added to any node that already has an EV charging station LATER
        loc_list = random.sample(power_nodes_list, self.num_charging_sites)  # randomization of charging locations

        for i in range(self.num_charging_sites):
            controller = control.MPC(
                self.controller_config)  # need to change this to load based on the users controller python file?
            # TODO: any controller object must have an action method which outputs a current for the current timestep
            # change config to false for centralized battery
            self.charging_config["locator_index"], self.charging_config["location"] = i, loc_list[i]
            charging_station = ChargingStation(battery, self.charging_config,
                                               controller)  # add controller and battery to charging station
            self.charging_sites[loc_list[i]] = charging_station
        self.stations_list = list(self.charging_sites.values())
        self.charging_locs = list(self.charging_sites.keys())
        self.storage_locs = list(self.storage_sites.keys())
        print(f"There are {len(self.battery_objects)} battery objects initialized")

    def initialize_aging_sim(self):
        # TODO: make the number of steps a passed in variable
        num_steps = 1
        self.aging_sim = BatterySim(0, num_steps)

    def initialize_price_loader(self, month):
        """Can add option for each charging site to have its own price loader"""
        configs_path = f'{self.path_prefix}/charging_sim/configs'
        current_working_dir = os.getcwd()
        self.price_loader = PriceLoader(self.prices_config, path_prefix=self.path_prefix)
        self.price_loader.set_month_data(month)
        input_data_res = self.prices_config["resolution"]
        if input_data_res > self.resolution:
            self.price_loader.downscale(input_data_res, self.resolution)
            self.prices_config["resolution"] = self.resolution
            file_path_list = self.prices_config["data_path"].split("_")
            new_data_path = self.prices_config["data_path"].replace(file_path_list[-1],
                                                                    str(self.resolution) + "min.csv")
            self.prices_config["data_path"] = new_data_path
            # print(new_data_path)
            with open(self.prices_config["config_path"], 'w') as config_file_path:
                os.chdir(configs_path)
                json.dump(self.prices_config, config_file_path, indent=1)
            os.chdir(current_working_dir)

    def reset_loads(self):
        self.site_net_loads = []

    def get_charging_sites(self):
        return self.charging_locs

    def get_storage_sites(self):
        return self.storage_locs

    def get_charger_obj_by_loc(self, loc):
        return self.charging_sites[loc]

    def get_storage_obj_by_loc(self, loc):
        return self.storage_sites[loc]

    def setup(self, power_nodes_list):
        self.create_charging_stations(power_nodes_list)
        self.initialize_price_loader(self.prices_config["month"])
        self.initialize_aging_sim()  # Battery aging

    def update_site_loads(self, load):
        self.site_net_loads += load,

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
        raise NotImplementedError("This method has not been implemented or will soon be deprecated!")

    def step(self, num_steps):
        # NEED TO ADD STEPPING THROUGH DAYS # should this be done in some controller master sim?
        # Full day prediction is not changing but the price is changing!! issues - is this fixed?
        """Propagates the simulation one step forward. Run controller and take one time-step action.."""
        #   FOR CENTRALIZED, LET'S USE ONLY ONE BATTERY FOR NOW AND USE THE AGGREGATE POWER FROM ALL STATIONS
        self.reset_loads()  # reset the loads from old time-step
        overall_charging_load = np.zeros((96, 1))
        elec_price_vec = self.price_loader.get_prices(self.time, self.num_steps) * 10  # need to freeze daily prices
        buffer_battery = self.battery_objects[0]  # first version only has one battery
        if self.time % 96 == 0:
            elec_price_vec = self.price_loader.get_prices(self.time, self.num_steps)
            self.time = 0  # reset time
            self.day_year_count = 0  # bug..finish this tonight!!
        for charging_station in self.stations_list:  # TODO: how can this be efficiently parallelized ?
            if self.time % 96 == 0:
                charging_station.controller.reset_load()
            self.control_start_index = num_steps * self.day_year_count
            stop = self.day_year_count + 14  # get today's load from test data; move to load generator; starting from the 14th index which is two weeks of history already happened
            self.control_shift = 0
            todays_load = self.test_data[stop]
            assert todays_load.size == 96
            todays_load.shape = (todays_load.size, 1)
            for _ in range(num_steps):
                # I need to abstract the entire controller away!
                predicted_load = charging_station.controller.predict_load(self.control_start_index,
                                                                          self.control_shift, stop)
                # to update MPC last observed load
                net_load = todays_load[self.time, 0]
                charging_station.update_load(net_load)  # set current load for charging station # UPDATED 6/8/22
                self.update_site_loads(net_load)  # Global Load Monitor for all the loads for this time-step
            overall_charging_load += predicted_load
            self.control_shift = 0
            # check whether one year has passed
            if self.day_year_count % 365 == 0:  # it has already run 365 days (This is wrt sampling Solar Generation)
                self.day_year_count = 0
                buffer_battery.start = 0

        control_action = self.central_storage_controller.compute_control(elec_price_vec, overall_charging_load)
        buffer_battery.dynamics(control_action)  # this should be aggregated
        buffer_battery.update_capacity()  # update the linear aging vector to include linear aging for previous run
        self.aging_sim.run(buffer_battery)  # simulate battery aging
        buffer_battery.start += 1
        self.time += 1
        return self.site_net_loads

    def load_results_summary(self):
        # TODO: selecting option for desired statistics
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        charging_sites_keys = self.charging_sites.keys()
        option1 = "loads"
        option2 = "storage"
        charging_stations = self.charging_sites.values()  # loads the charging stations

        for charging_station in charging_stations:
            charging_station.save_sim_data()
            # charging_station.visualize(option=option1)
        for battery in self.battery_objects:
            battery.visualize("SOC_track")
            battery.visualize("voltages")
            battery.visualize("true_aging")
            battery.visualize("SOH_track")

        # SOME ADDITIONAL PLOTS BELOW
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(battery.SOC_track[1:], label='battery SoC')
        ax2.plot(battery.calendar_aging[1:], color='r', ls='--', label='calendar aging')
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('SOC')
        ax2.set_ylabel('Calendar aging')
        plt.savefig("Calendar_SOC_plot.png")
        plt.close('all')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(battery.SOC_track[1:], label='battery SOC')
        ax2.plot(battery.cycle_aging[1:], color='r', ls='--', label='cycle aging')
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('SOC')
        ax2.set_ylabel('Cycle aging')
        ax1.legend()
        ax2.legend()
        plt.savefig("Cycle_SOC_plot.png")
        plt.close('all')

        print(f"total calendar aging is {sum(battery.calendar_aging)}")
        print(f"total cycle aging is {sum(battery.cycle_aging)}")
