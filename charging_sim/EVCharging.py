from chargingStation import ChargingStation
print('station')
import numpy as np
from config import energy_prices_TOU, add_power_profile_to_object, show_results, solar_gen
print('ok')
from plots import plot_results
print('plots done')
from battery import Battery
print('battery')
from batteryAgingSim import BatterySim
print('aging done')
from controller import MPC
print('controller done')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class ChargingSim:
    def __init__(self, num_charging_sites):

        data2015 = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2015.csv')  # Need to update these dirs.
        data2016 = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2016.csv')
        data2017 = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2017.csv')
        data2018 = np.genfromtxt('/home/ec2-user/EV50_cosimulation/CP_ProjectData/power_data_2018.csv')
        charge_data = np.vstack([data2015[7:, ], data2016, data2017])/10
        test_data = data2018[:-1, ]/10  # removing bad data
        self.charge_data = charge_data
        self.battery_specs_per_loc = None   # Could be used later to specify varying batteries for various nodes
        self.day_year_count = 0
        self.stop = 0
        self.control_start_index = 0
        self.control_shift = 0
        self.test_data = test_data
        self.num_charging_sites = num_charging_sites
        self.charging_locs = []
        self.stations_list = []
        self.battery_objects = []
        self.site_net_loads = []
        self.aging_sim = None # This gets updated later
        y = self.test_data
        self.std_y = np.std(y, 0)
        self.std_y[self.std_y == 0] = 1
        self.mean_y = np.mean(y, 0)
        self.scaled_test_data = (y - self.mean_y)/self.std_y  # network output
        self._nodes = []

        scaler = MinMaxScaler()
        onestep_data = np.reshape(charge_data, (charge_data.size, 1))
        scaler.fit(onestep_data)
        self.scaled_test_data_onestep = [scaler.transform(np.reshape(test_data, (test_data.size, 1))), scaler]

    def create_battery_objects(self):
         # this stores all battery objects in the network
        Q_initial = 3.5
        for idx in range(self.num_charging_sites):
            buffer_battery = Battery("Tesla Model 3", Q_initial)
            buffer_battery.track_SOC(4)  # does not do anything for now
            self.battery_objects.append(buffer_battery)
            buffer_battery.id = idx
            buffer_battery.num_cells = buffer_battery.battery_setup(voltage=375, capacity=8000,
                                                     cell_params=(buffer_battery.nominal_voltage, 4.85))

    def create_charging_stations(self):
        loc_list = list(range(self.num_charging_sites))  # This will be obtained from Lily's dist models.
        print(len(self.battery_objects))
        print(loc_list)
        power_cap = 100  # kW
        for i in range(self.num_charging_sites):
            print(i)
            charging_station = ChargingStation(self.battery_objects[i], loc_list[i], power_cap, i)
            self.stations_list.append(charging_station)

    def initialize_aging_sim(self):
        # TODO: make the number of steps a passed in variable
        self.aging_sim = BatterySim(0, 96)

    def reset_loads(self):
        self.site_net_loads = []

    def get_charging_sites(self):
        return self.charging_locs

    def setup(self):
        self.create_battery_objects()
        self.create_charging_stations()
        self.initialize_aging_sim()

    def step(self, num_steps):
        """Step forward once. Run MPC controller and take one time-step action.."""
        self.reset_loads()  # reset the loads from old time-step
        for charging_station in self.stations_list:
            buffer_battery = charging_station.storage
            control = MPC(self.charge_data, self.scaled_test_data,
                          self.scaled_test_data_onestep, buffer_battery, self.std_y, self.mean_y)
            total_savings = 0
            run_count = 0
            day_year_count = 0
            EOL = 0.8 * buffer_battery.get_properties()["energy_nom"]  # End of life battery capacity
            controls = []
            start_day = 0  # states where to begin sampling past data
            self.control_start_index = num_steps * start_day
            stop = start_day + 14  # get today's load from test data
            self.control_shift = 0
            todays_load = self.test_data[stop]
            assert todays_load.size == 96
            todays_load.shape = (96, 1)
            loads = []
            for i in range(num_steps):
                control_action, predicted_load = control.compute_control(self.control_start_index,
                                                                         self.control_shift, stop, buffer_battery.size)
                loads.append(predicted_load[i, 0])
                control.load = np.append(control.load, todays_load[i])  # update load with the true load, not prediction,
                # to update MPC last observed load
                controls.append(control_action[0] - control_action[1])
                net_load = todays_load - (control_action[0] - control_action[1])
                self.control_start_index += 1
                self.control_shift += 1
                self.site_net_loads.append(net_load) # Global Load Monitor for all the loads for this time-step
                print("site net loads shape is: ", len(self.site_net_loads))

            stop += 1  # shifts to the next day
            self.control_shift = 0
            print("MSE is ", (np.average(control.full_day_prediction - todays_load)**2)**0.5, "for day ", stop)
            if self.control_shift == 96:
                control.reset_load()
                self.stop += 1

            # update season based on run_count (assumes start_time was set to 0)
            run_count += 1
            print("Number of cycles is {}".format(run_count))
            # total_savings = show_results(total_savings, buffer_battery, energy_prices_TOU, todays_load)

            # check whether one year has passed
            if self.day_year_count % 365 == 0:  # it has already run for 365 days (This is wrt sampling Solar Generation)
                self.day_year_count = 0
                buffer_battery.start = 0

            buffer_battery.update_capacity()  # updates the capacity to include aging for previous run
            self.aging_sim.run(buffer_battery)  # run battery response to actions\
            buffer_battery.Q_initial = buffer_battery.Q.value[1][0]  # this is for the entire battery
            # plot_results(todays_load, buffer_battery, solar_gen[buffer_battery.start:buffer_battery.start + num_steps])
            start_time = buffer_battery.start
            print(num_steps, start_time)
            print(todays_load.shape, buffer_battery.power_charge.shape, buffer_battery.power_discharge.shape, solar_gen[start_time:start_time + num_steps].shape)
            EV_power_profile = todays_load[0:num_steps+1,] + buffer_battery.power_charge.value[0:num_steps+1,] - \
                                 buffer_battery.power_discharge.value[0:num_steps+1,] - solar_gen[start_time:start_time + num_steps]
            add_power_profile_to_object(buffer_battery, day_year_count, EV_power_profile)  # update power profile
            print("SOH is: ", buffer_battery.SOH)
        self.day_year_count += 1
        plt.plot(buffer_battery.track_true_aging)
        plt.plot(np.array(buffer_battery.track_linear_aging)/0.2)
        plt.title("true aging and Linear aging")
        plt.legend(["True aging", "Linear aging"])
        plt.show()
        return self.site_net_loads

