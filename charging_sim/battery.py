# import cvxpy as cp
import json
import numpy as np
from utils import num_steps
import matplotlib.pyplot as plt

# OCV_SOC_linear_params = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/OCV_SOC_linear_params_NMC_25degC.npy')

# Need to understand charging dynamics as well. Cannot assume symmetry

# TODO: add battery simulation resolution control so can be different from control as well...
class Battery:
    """
     Properties:
         max voltage(V)
         min Voltage (V)
         nominal energy (kWh)
         id (-)
         ambient temperature (C)

     Assumptions:
         efficiency already incorporates resistance; improve later.
         assumes linear aging model for SOH, SOC, and voltage variation, but relaxed by parameter update.
         Temperature variation in battery environment is negligible.

    """

    def __init__(self, battery_type, Q_initial, node=None, config=None):
        """should I make in terms of rc pairs or how should I call dynamics..data-driven vs ECM"""
        self._node = node  # This is used for battery location.
        #   TODO: need to include resolution and others
        self.resolution = config["resolution"]
        self.dt = config["resolution"] / 60     # in hours
        self.cap = config["cell_nominal_cap"]  # Ah
        self.nominal_cap = config["cell_nominal_cap"]
        self.cell_resistance = config["resistance"]  # TO be updated
        self._eff = config["round-trip_efficiency"]
        self.max_c_rate = config["max_c_rate"]
        # self.nominal_energy = config["cell_nominal_energy"]     # Watts-hours

        # LOAD THE PARAMETERS FOR CERTAIN CYCLE INTERVAL
        self.params_0_cycles = config["params_0_cycles"]
        self.params_25_cycles = config["params_25_cycles"]
        self.params_75_cycles = config["params_75_cycles"]
        self.params_125_cycles = config["params_125_cycles"]
        # TODO: think about how to load these parameter changes over time
        self.OCV_map_SOC = config["OCV_map_SOC"]   # update these two as well
        self.OCV_map_voltage = config["OCV_map_voltage"]   # update these two as well

        self.max_current = self.max_c_rate * self.cap   # figure out a way to have these update automatically
        self.A_Ro = self.params_0_cycles[0]
        self.B_Ro = self.params_0_cycles[1]
        self.C_Ro = self.params_0_cycles[2]
        self.R1 = self.params_0_cycles[3]
        self.C1 = self.params_0_cycles[4]
        self.R2 = self.params_0_cycles[5]
        self.C2 = self.params_0_cycles[6]

        self._name = battery_type
        self.iR1 = 0
        self.iR2 = 0

        self.voltage = config["max_cell_voltage"]
        self.max_voltage = config["max_cell_voltage"]
        self.min_voltage = config["min_cell_voltage"]  # for each cell
        self.nominal_voltage = config["nominal_cell_voltage"]  # for each cell
        self.nominal_energy = self.nominal_voltage * self.nominal_cap
        self.Qmax = config["SOC_max"] * self.cap
        self.Qmin = config["SOC_min"] * self.cap
        self.initial_SOC = config["SOC"]
        self.min_SOC = config["SOC_min"]
        self.max_SOC = config["SOC_max"]
        self.OCV = config["max_cell_voltage"]
        self.SOH = config["SOH"]
        self.daily_self_discharge = config["daily_self_discharge"]

        # battery_setup updates with tuple (no_cells_series, no_modules_parallel, total_cells)
        self.SOC_track = [self.initial_SOC]
        self.SOH_track = [self.SOH]     # to be finished later
        self.control_power = np.array([])
        # self.OCV = cp.Variable((num_steps, 1))
        # self.discharge_current = cp.Variable((num_steps, 1))  # current
        # self.discharge_voltage = cp.Variable((num_steps, 1))
        self.voltages = []  # to be updated at each time-step (seems expensive)
        self.current_voltage = 0
        self.true_voltage = np.array([])  # discharge/charge voltage per time-step
        # self.Q = cp.Variable((num_steps + 1, 1))  # Amount of energy Kwh available in battery
        self.SOC = self.initial_SOC
        self.SOC_list = [self.initial_SOC]

        self.Q_initial = Q_initial  # include the units here
        self.control_current = np.array([])
        self.total_amp_thruput = 0

        # self.power_charge = cp.Variable((num_steps, 1))
        # self.power_discharge = cp.Variable((num_steps, 1))
        self.power = 0
        self.current = 0
        self.true_power = []
        self.start = config["start_time"]
        self.MPC_Control = {'Q': [], 'P': []}  # tracks the control actions for the MPC control
        self.size = 100  # what does this mean?
        self.cell_count = 0

        self.total_aging = 0
        self.true_capacity_loss = 0
        self.resistance_growth = 0
        # self.true_SOC = np.empty((num_steps + 1, 1))  #  BatterySim updates this with the true SOC different from Optimization
        self.true_aging = []  # want to keep track to observe trends based on degradation models
        self.linear_aging = []  # linear model per Hesse Et. Al

        self.ambient_temp = 25  # Celsius
        self.charging_costs = 0
        self.power_profile = {'Jan': [], 'Feb': [], 'Mar': [], 'Apr': [], 'May': [], 'Jun': [], 'Jul': [], 'Aug': [],
                              'Sep': [], 'Oct': [], 'Nov': [], 'Dec': []}
        # self.Ro = self.get_Ro()

        self.topology = None
        self.constraints = None
        self.id = None  # Use Charging Station ID for this
        self.savings = None
        self.state_of_charge = None
        self.nominal_pack_voltage = None    # to be initialized later
        self.nominal_pack_cap = None
        self.predicted_voltages = list()
        self.operating_voltages = list()

    def get_true_power(self):
        return np.array(self.true_power)

    def battery_setup(self, pack_voltage, capacity, cell_params):
        """
        TODO: I NEED TO CHECK THAT THE PACK MATCHES THE OUTPUT POWER CORRECTLY
        Capacity (Wh)
        Voltage (V)
        params: (cell_amp_hrs (Ah), cell_voltage (V)
        TODO: Finish this up and scale battery up completely.
            Do comparison algo setup with real battery and parameter adjustment."""
        # number of modules in parallel should be determined by the power rating and Voltage
        # should use nominal voltage and max allowable current
        cell_amp_hrs = cell_params[0]
        cell_voltage = cell_params[1]
        no_cells_series = pack_voltage // cell_voltage
        no_modules_parallel = capacity // cell_amp_hrs
        self.cell_count = no_cells_series * no_modules_parallel
        self.topology = (no_cells_series, no_modules_parallel, self.cell_count)
        self.nominal_pack_voltage = no_cells_series * self.nominal_voltage
        self.nominal_pack_cap = no_modules_parallel * self.cap
        print("***** Battery initialized. *****\n",
              "Capacity is {} Ah".format(capacity),
              "Total number of cells is: {} .".format(self.cell_count))
        return self.cell_count

    def battery_setup_tesla(self, model=3):
        """Using TESLA mode to setup battery config"""
        if model == 3:
            pass
        elif model == 's':
            pass

    def est_calendar_aging(self):
        """Estimates the constant calendar aging of the battery. this is solely time-dependent.
        Deprecate this later for this object"""
        life_cal_years = 10
        seconds_in_min = 60
        seconds_in_year = 31556952
        aging_cal = (self.resolution * seconds_in_min) / (life_cal_years * seconds_in_year)
        aging_cal *= np.ones((num_steps, 1))
        return np.sum(aging_cal)

    def est_cyc_aging(self):
        """Creates linear battery ageing model per hesse. et al, and returns its cvx object.
        Deprecate this later for this object"""
        seconds_in_min = 60
        life_cyc = 4500     # change this to be input in the config file
        aging_cyc = (0.5 * (np.sum(abs(self.current * self.voltage))
                            * self.resolution / seconds_in_min)) / (life_cyc / 0.2 * self.nominal_energy)
        return aging_cyc

    def get_power_profile(self, months):
        power_profiles_dict = {}
        for month in months:
            power_profiles_dict[month] = self.power_profile[month]
        return power_profiles_dict

    def get_total_aging(self):
        return self.est_cyc_aging() + self.est_calendar_aging()

    def get_aging_value(self):
        """returns the actual aging value lost after a cvxpy run..."""
        aging_value = self.est_cyc_aging() + self.est_calendar_aging()
        return aging_value

    def update_capacity(self):
        """This is not true capacity but anticipated capacity based on linear model."""
        aging_value = self.get_aging_value() * 0.2  # do not multiply by 0.2 to keep comparable
        self.total_aging += aging_value
        self.linear_aging.append(aging_value)  # this is to ADD aging for each time step

    def get_Ro(self):
        """This can be updated later to include current...
        Deprecate this possibly"""
        pass
        # Ro = self.ECM_params[1] * cp.exp(self.SOC) + self.ECM_params[0] * cp.exp(self.C_Ro * self.SOC) \
        #      + self.resistance_growth
        # return Ro

    def update_SOC(self):
        action_length = 1   # maybe change this later for variable action length
        self.Q_initial = self.Q.value[action_length]

    def track_SOC(self, SOC):
        self.SOC_track.append(SOC)
        self.SOC_list.append(SOC)

    def update_max_current(self, verbose=False):
        self.max_current = self.max_c_rate * self.cap
        if verbose:
            print("Maximum allowable current updated.")

    def update_voltage(self, voltage):
        self.current_voltage = voltage  # I should be updating initial voltage with new voltage measurement
        self.predicted_voltages.append(voltage)
        print("Current voltage estimate is: ", voltage)

    def get_properties(self):
        return self.properties

    def visualize(self, option):
        if type(option) == str:
            plt.style.use('seaborn-darkgrid')
            if option == "SOC_track":
                plotting_values = getattr(self, option)
                plt.figure()
                plt.plot(plotting_values)
                plt.xlabel("Time Step")
                plt.ylabel("SOC")
                plt.savefig(option + "_{}.png".format(self.id))
                plt.close()
            else:
                plotting_values = getattr(self, option)
                plt.figure()
                plt.plot(plotting_values)
                plt.xlabel("Time Step")
                plt.ylabel(option)
                plt.savefig(option + "_{}.png".format(self.id))
                plt.close()


    def visualize_voltages(self):
        pass

    def learn_params(self):
        pass

    def update_params(self):
        """This changes the battery params depending on the number of cycles"""
        pass
    
    def dynamics(self, current):
        # currently, dynamics assumes cells are perfectly balanced- Can we account for imbalanced cells later?
        # TODO: deal with simulation dual-resolution dynamical system...maybe get finer battery dynamics in here

        dt = self.dt * 3600  # convert from hour to seconds for dynamics equations but not SOC
        # state equations
        self.SOC = self.SOC + current * self.dt / self.cap   # current signs (+) charge (-) discharge
        # For a particular step, if absolute val Crate is so high that SOC exceeds limits, skip dynamics
        if self.SOC >= self.max_SOC:
            excess_soc = self.SOC - self.max_SOC
            allowable_curr_thruput = abs(current) * self.dt - excess_soc * self.cap    # Ah
            assert allowable_curr_thruput >= 0  # remove this later
            self.total_amp_thruput += allowable_curr_thruput
            print('SOC violation, readjusting SoC...')
            self.SOC = self.max_SOC
            self.track_SOC(self.SOC)
            self.voltage = self.max_voltage
            self.voltages = np.append(self.voltages, self.voltage)
            return self.voltage
        if self.SOC <= self.min_SOC:
            print('SOC violation, readjusting SoC...')
            excess_soc = self.min_SOC - self.SOC
            allowable_curr_thruput = abs(current) * self.dt - excess_soc * self.cap  # Ah
            assert allowable_curr_thruput >= 0
            self.total_amp_thruput += allowable_curr_thruput
            self.voltage = self.min_voltage
            self.SOC = self.min_SOC
            self.track_SOC(self.SOC)
            self.voltages = np.append(self.voltages, self.voltage)
            return self.voltage

        self.OCV = np.interp(self.SOC, self.OCV_map_SOC, self.OCV_map_voltage)  # change this to OCV map
        # print("OCV is : ", self.OCV, "SOC is: ", self.SOC)
        # print(self.SOC, self.OCV_map_SOC[0], self.OCV_map_voltage[0], self.OCV_map_SOC[-1], self.OCV_map_voltage[-1])
        # plt.plot(self.OCV_map_SOC, self.OCV_map_voltage)
        # plt.savefig('OCV_map')
        # plt.close()
        self.Ro = self.B_Ro * np.exp(self.SOC) + self.A_Ro * np.exp(self.C_Ro * self.SOC)

        #   state equations
        self.iR1 = np.exp(-dt / (self.R1 * self.C1)) * self.iR1 + (1 - np.exp(-dt / (self.R1 * self.C1))) * current
        self.iR2 = np.exp(-dt / (self.R2 * self.C2)) * self.iR2 + (1 - np.exp(-dt / (self.R2 * self.C2))) * current
        self.voltage = self.OCV + current * self.Ro + self.iR1 * self.R1 + self.iR2 * self.R2
        if self.voltage > self.max_voltage:
            self.voltage = self.max_voltage
            print('max voltage exceeded...adjusting')   # I don't think this should happen, so FLAG if it does
            self.voltages = np.append(self.voltages, self.voltage)  # numpy array
            self.track_SOC(self.SOC)
            return self.voltage
        if self.voltage < self.min_voltage:
            print('min voltage reached...overdischarge', self.voltage)
            self.voltage = self.min_voltage

            self.voltages = np.append(self.voltages, self.voltage)  # numpy array
            self.track_SOC(self.SOC)
            return self.voltage

        self.current = current
        self.power = (self.voltage * self.topology[0]) * (self.current * self.topology[1]) / 1000  # kw
        print("power is {} kW".format(self.power))
        # self.topology[0] = # of cells in series, self.topology[1] = # in parallel
        self.voltages = np.append(self.voltages, self.voltage)  # numpy array
        self.track_SOC(self.SOC)
        self.total_amp_thruput += current * self.dt
        return self.voltage

#   TEST THE BATTERY CODE HERE
def test():
    battery_config_path = "/home/ec2-user/EV50_cosimulation/charging_sim/configs/battery.json"
    with open(battery_config_path, "r") as f:
        battery_config = json.load(f)
    params_list = [key for key in battery_config.keys() if "params_" in key]
    for params_key in params_list:
        print("testing load battery params: ", params_key)
        battery_config[params_key] = np.loadtxt(battery_config[params_key])
    # do the OCV maps as well
    battery_config["OCV_map_voltage"] = np.loadtxt(battery_config["OCV_map_voltage"])[::-1]     # ascending order
    battery_config["OCV_map_SOC"] = np.loadtxt(battery_config["OCV_map_SOC"])[::-1]     # ascending order

    Q_initial = 3.5
    buffer_battery = Battery("Tesla Model 3", Q_initial, config=battery_config)
    buffer_battery.id, buffer_battery.node = 0, 0

    # test dynamics here
    c = -4.85
    voltages = []
    for i in range(500):
        v = buffer_battery.dynamics(c)
        voltages.append(v)
    c = 0
    for i in range(500):
        v = buffer_battery.dynamics(c)
        voltages.append(v)
    c = -2.8
    for i in range(100):
        v = buffer_battery.dynamics(c)
        voltages.append(v)
    c = -4.85
    for i in range(20):
        v = buffer_battery.dynamics(c)
        voltages.append(v)
    c = 0
    for i in range(400):
        v = buffer_battery.dynamics(c)
        voltages.append(v)
    c = 4.2
    for i in range(100):
        v = buffer_battery.dynamics(c)
        voltages.append(v)
    plt.plot(voltages)
    plt.xlabel(" Time Step (seconds) ")
    plt.ylabel(" Voltage ")
    plt.savefig("battery_test_plot")
    plt.close()


if __name__ == "__main__":
    test()
