import json
import numpy as np
from utils import num_steps
import matplotlib.pyplot as plt
import os


# Model assumes symmetry from charging to discharging dynamics. Asymmetry is negligible for most purposes.


# TODO: add battery simulation resolution control so can be different from control as well...
#  ALso track the number of voltage control violations during control computations in the future

class Battery:
    """
     Properties:
        properties are mainly controlled in the battery config file 'battery.json'
         max-c-rate determines the power capacity of the cell as a multiple of the value of the energy capacity
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

    def __init__(self, battery_type=None, node=None, config=None, controller=None):
        """should I make in terms of rc pairs or how should I call dynamics..data-driven vs ECM"""
        self.node = node  # This is used for battery location.
        #   TODO: need to include resolution and others
        self.controller = controller
        self.resolution = config["resolution"]
        self.dt = config["resolution"] / 60  # in hours
        self.cap = config["cell_nominal_cap"]  # Ah
        self.cell_nominal_cap = config["cell_nominal_cap"]
        self.R_cell = config["resistance"]  # single cell resistance
        self.R_pack = None  # overall pack resistance
        self._eff = config["round-trip_efficiency"]
        self.max_c_rate = config["max_c_rate"]
        self.battery_cost = 200  # this is in $/kWh
        self.To = 273.15 + 23  # ambient temperature   (USE THE SAME INPUT AS TRANSFORMER AGING MODEL
        self.T = self.To  # initialize at atmposheric temperature
        # self.nominal_energy = config["cell_nominal_energy"]     # Watts-hours

        # LOAD THE PARAMETERS FOR CERTAIN CYCLE INTERVAL
        self.params_0_cycles = config["params_0_cycles"]
        self.params_25_cycles = config["params_25_cycles"]
        self.params_75_cycles = config["params_75_cycles"]
        self.params_125_cycles = config["params_125_cycles"]
        # TODO: think about how to load these parameter changes over time
        self.OCV_map_SOC = config["OCV_map_SOC"]  # update these two as well
        self.OCV_map_voltage = config["OCV_map_voltage"]  # update these two as well

        self.max_current = self.max_c_rate * self.cap  # figure out a way to have these update automatically
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

        self.max_voltage = config["max_cell_voltage"]
        self.pack_max_voltage = config["pack_max_voltage"]
        self.min_voltage = config["min_cell_voltage"]  # for each cell
        self.pack_max_Ah = config["pack_max_Ah"]  # this is used in battery_setup_2 to scale voltage instead of curr
        self.pack_nom_Ah = config["pack_nom_Ah"]
        self.cell_nominal_voltage = config["nominal_cell_voltage"]  # for each cell
        self.nominal_energy = self.cell_nominal_voltage * self.cell_nominal_cap
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
        self.predicted_SOC = [self.initial_SOC]
        self.pred_power = [0]
        self.SOH_track = [self.SOH]  # to be finished later
        self.calendar_aging = [0.0]  # tracking calendar aging
        self.cycle_aging = [0.0]  # tracking cycle aging
        self.control_power = np.array([])

        self.voltage = np.interp(self.initial_SOC, self.OCV_map_SOC, self.OCV_map_voltage)  # this is wrong
        self.voltages = [self.voltage]  # to be updated at each time-step (seems expensive)
        self.current_voltage = 0.0
        self.true_voltage = np.array([])  # discharge/charge voltage per time-step
        # self.Q = cp.Variable((stepsize + 1, 1))  # Amount of energy Kwh available in battery
        self.SOC = self.initial_SOC
        self.SOC_list = [self.initial_SOC]
        self.Ro = self.B_Ro * np.exp(self.SOC) + self.A_Ro * np.exp(self.C_Ro * self.SOC)  # optional

        self.Q_initial = 0  # include the units here
        self.control_current = []  # changed to a list - will be more efficient
        self.total_amp_thruput = 0.0
        self.currents = [0]

        self.power = 0
        self.current = 0
        self.true_power = [0]
        self.start = config["start_time"]
        self.MPC_Control = {'Q': [], 'P': []}  # tracks the control actions for the MPC control
        self.size = 100  # what does this mean?...I should use this for accounting how much power I CAN DEMAND
        self.cell_count = 0

        self.total_aging = 0
        self.true_capacity_loss = 0
        self.resistance_growth = 0
        self.true_aging = []  # want to keep track to observe trends based on degradation models
        self.linear_aging = []  # linear model per Hesse Et. Al

        self.ambient_temp = 25  # Celsius (NOT USED YET - FUTURE VERSION COULD USE THIS)
        self.charging_costs = 0
        self.power_profile = {'Jan': [], 'Feb': [], 'Mar': [], 'Apr': [], 'May': [], 'Jun': [], 'Jul': [], 'Aug': [],
                              'Sep': [], 'Oct': [], 'Nov': [], 'Dec': []}
        # self.Ro = self.get_Ro()

        self.topology = [1, 1]  # DEFAULT VALUE: singular cell
        self.constraints = None
        self.id = None
        self.location = None
        self.savings = None
        self.state_of_charge = None
        self.nominal_pack_voltage = config["pack_voltage"]  # to be initialized later
        self.pack_energy_capacity = config["pack_energy_cap"]  # battery rating this is in Watt-hours (Wh)
        self.nominal_pack_cap = None  # will be set in battery setup
        self.predicted_voltages = []
        self.operating_voltages = []

    def get_true_power(self):
        return np.array(self.true_power)

    def battery_setup(self):
        """
        Capacity (Wh)
        Voltage (V)
        params: (cell_amp_hrs (Ah), cell_voltage (V)"""
        # number of modules in parallel should be determined by the power rating and Voltage
        # should use nominal voltage and max allowable current
        print(f"**** Pre-initialized nominal pack voltage is {self.nominal_pack_voltage}")
        pack_capacity_Ah = self.pack_energy_capacity / self.pack_max_voltage
        cell_amp_hrs = self.cell_nominal_cap  # for a cell (Ah) Maximum
        no_cells_series = round(self.pack_max_voltage / self.max_voltage)  # cell nominal
        no_modules_parallel = round(pack_capacity_Ah / (cell_amp_hrs + 1e-8))
        self.cell_count = no_cells_series * no_modules_parallel
        self.topology = (no_cells_series, no_modules_parallel, self.cell_count)
        self.nominal_pack_voltage = no_cells_series * self.cell_nominal_voltage
        self.nominal_pack_cap = no_modules_parallel * self.cap
        self.R_cell = self.Ro + self.R1 + self.R2
        # self.cell_capacitance
        series_resistance = no_cells_series * self.R_cell
        # series_capacitance = 1/(no_cells_series * self.C1 )
        self.R_pack = 1 / (no_modules_parallel * 1 / series_resistance)
        print(f"**** Post-initialized nominal pack voltage is {self.nominal_pack_voltage}")
        print("***** Battery initialized. *****\n",
              f"Battery pack capacity is {pack_capacity_Ah} Ah",
              f"Battery pack resistance is {self.R_pack} Ohm",
              f"Total number of cells is: {self.cell_count} .\n",
              f"no. cells in series is: {no_cells_series} \n. No modules in parallel is: {no_modules_parallel}"
              )

    def battery_setup_2(self):
        """
        This will scale up voltage instead of current capacity (Ah).
        Capacity (Wh)
        Voltage (V)
        params: (cell_amp_hrs (Ah), cell_voltage (V)"""
        # number of modules in parallel should be determined by the power rating and Voltage
        # should use nominal voltage and max allowable current
        # print("**** Pre-initialized nominal pack voltage is {}".format(self.nominal_pack_voltage))
        self.pack_max_voltage = self.pack_energy_capacity / self.pack_nom_Ah
        no_cells_series = round(self.pack_max_voltage / self.max_voltage)  # cell nominal
        no_modules_parallel = round(self.pack_max_Ah / (self.cell_nominal_cap + 1e-8))
        self.cell_count = no_cells_series * no_modules_parallel
        self.topology = (no_cells_series, no_modules_parallel, self.cell_count)
        self.nominal_pack_voltage = no_cells_series * self.cell_nominal_voltage
        self.R_cell = self.Ro + self.R1 + self.R2
        series_resistance = no_cells_series * self.R_cell
        self.R_pack = 1 / (no_modules_parallel * 1 / series_resistance)
        print(f"**** Post-initialized nominal pack voltage is {self.nominal_pack_voltage}")
        print("***** Battery initialized. *****\n",
              f"Battery pack capacity is {self.pack_max_Ah} Ah",
              f"Battery pack resistance is {self.R_pack} Ohm",
              f"Total number of cells is: {self.cell_count} .\n",
              f"no. cells in series is: {no_cells_series} \n. No modules in parallel is: {no_modules_parallel}"
              )

    def battery_setup_tesla(self, model=3):
        """Using TESLA mode to setup battery config. TO BE IMPLEMENTED LATER"""
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
        life_cyc = 4500  # change this to be input in the config file
        return (0.5 * (np.sum(abs(self.current * self.voltage)) * self.resolution / seconds_in_min)) / \
            (life_cyc / 0.2 * self.nominal_energy)

    def get_power_profile(self, months):
        return {month: self.power_profile[month] for month in months}

    def get_total_aging(self):
        return self.est_cyc_aging() + self.est_calendar_aging()

    def get_aging_value(self):
        """returns the actual aging value lost after a cvxpy run..."""
        return self.est_cyc_aging() + self.est_calendar_aging()

    def update_capacity(self):
        """This is not true capacity but anticipated capacity based on linear model."""
        aging_value = self.get_aging_value() * 0.2  # do not multiply by 0.2 to keep comparable
        self.total_aging += aging_value
        self.linear_aging += aging_value,
        self.update_max_current()

    def get_Ro(self):
        """This can be updated later to include current...
        Deprecate this possibly"""
        pass

    def get_roundtrip_efficiency(self):
        return self._eff

    def track_SOC(self, SOC):
        self.SOC_track += SOC,
        self.SOC_list += SOC,

    def update_max_current(self, verbose=False):
        # self.max_current = self.max_c_rate * self.cap * self.topology[1]
        if verbose:
            print("Maximum allowable current updated.")

    def update_voltage(self, voltage):
        self.current_voltage = voltage  # I should be updating initial voltage with new voltage measurement
        # self.predicted_voltages.append(voltage)
        self.predicted_voltages += voltage,
        # print("Current voltage estimate is: ", voltage)

    def visualize(self, option):
        """method is used to plot and save battery states desired by user"""
        if type(option) == str:
            plt.style.use('seaborn-darkgrid')
            plotting_values = getattr(self, option)
            if option == "SOC_track":
                plt.figure()
                plt.plot(plotting_values)
                plt.xlabel("Time Step")
                plt.ylabel("SOC")
                plt.savefig(f"{option}_{self.id}.png")
                print(f"Saving values for {self.id}")
            else:
                plt.figure()
                plt.plot(plotting_values)
                plt.xlabel("Time Step")
                plt.ylabel(option)
                plt.savefig(f"{option}_{self.id}.png")
            plt.close()
        print("Est. tot. no. of cycles is: ", self.total_amp_thruput / ((self.cell_nominal_cap + self.cap) / 2),
              'cycles')

    def save_sim_data(self, save_prefix):
        """working on this, not tested yet"""
        import pandas as pd
        save_file_base = f'{str(self.id)}_{self.node}'
        data = {'SOC': self.SOC_track,
                'SOC_pred': self.predicted_SOC,
                'SOH': self.SOH_track,
                'Voltage_pack': np.array(self.voltages),
                'currents_pack': np.array(self.currents),
                'cycle_aging': np.array(self.cycle_aging),
                'calendar_aging': np.array(self.calendar_aging),
                'power_kW': np.array(self.true_power),
                'pred_power_kW': np.array(self.pred_power)}
        pd.DataFrame(data).to_csv(f'{save_prefix}/battery_sim_{save_file_base}.csv')
        total_cycles = 0.5 * self.total_amp_thruput / (self.topology[1] * (self.cell_nominal_cap + self.cap) / 2)
        np.savetxt(f'{save_prefix}/total_batt_cycles_{save_file_base}.csv', [total_cycles])
        np.savetxt(f'{save_prefix}/pack_resistance_{save_file_base}.csv', [self.R_pack])
        print('***** Successfully saved simulation outputs to: ', f'battery_sim_{save_file_base}.csv')
        print("Est. tot. no. of cycles is: ", total_cycles, 'cycles')

    def learn_params(self):
        pass

    def update_params(self):
        """This changes the battery params depending on the number of cycles"""
        pass

    def dynamics(self, current, simulate_pack=True):
        # currently, dynamics assumes cells are perfectly balanced- Can we account for imbalanced cells later?
        #  state equations
        self.state_eqn(current)  # this updates the battery states

        if self.voltage > self.max_voltage:
            print("charge current too high! Max voltage exceeded")
            # we de-rate the current if voltage is too high (exceeds max prescribed v)
            # voltage can exceed desirable range if c-rate is too high, even when SoC isn't at max
            current -= (self.voltage - self.max_voltage) / self.R_pack  # changed from just Ro
            self.voltage = self.max_voltage  # WHY AM I SETTING THE MAX VOLTAGE HERE INSTEAD OF JUST LETTING STATE EQN DETERMINE THE VALUE
            print("max testing voltage is: ", self.voltage)
            self.state_eqn(current, append=False)
            print("max testing voltage is: ",
                  self.voltage)  # when you come back, test and DOUBLE CHECK THIS. Getting closer to full simulation.
            self.currents[-1] = current
            self.power = self.max_voltage * self.current / 1000
            self.true_power[-1] = self.power
            self.SOC = self.SOC + current * self.dt / (self.cap * self.topology[1])
            self.voltages += self.max_voltage,  # numpy array
            self.track_SOC(self.SOC)
            self.total_amp_thruput += abs(current) * self.dt  # cycle counting
            return self.voltage
        elif self.voltage < self.min_voltage:
            print("discharge current too high ! Min voltage exceeded")
            current += (self.min_voltage - self.voltage) / self.R_cell
            self.state_eqn(current, append=False)
            self.currents[-1] = current
            self.power = self.voltage * self.current / 1000
            self.true_power[-1] = self.power
            self.voltage = self.min_voltage
            self.voltages += self.min_voltage,
            self.SOC = self.SOC + current * self.dt / (self.cap * self.topology[1])
            self.track_SOC(self.SOC)
            self.total_amp_thruput += abs(current) * self.dt  # cycle counting
            return self.voltage

        self.current = current
        self.SOC = self.SOC + current * self.dt / (self.cap * self.topology[1])
        self.voltages += self.voltage,  # numpy array
        self.track_SOC(self.SOC)
        self.total_amp_thruput += abs(current) * self.dt  # cycle counting
        return self.voltage

    def load_pack_props(self):
        # TODO: write a method that builds a pack from cells in any config
        # first get the series resistance and capacitance
        self.R1 *= self.topology[0]
        self.R2 *= self.topology[0]
        self.C1 /= self.topology[0]
        self.C2 /= self.topology[0]

        # now obtain the overall parallel resistance
        self.R1 /= self.topology[1]
        self.R2 /= self.topology[1]  # each new parallel path reduces the overall resistance
        self.C1 *= self.topology[1]
        self.C2 *= self.topology[1]
        self.R_pack = self.Ro + self.R1 + self.R2
        self.max_voltage = self.max_voltage * self.topology[0]
        self.max_current *= self.topology[1]
        self.voltages = [self.voltage * self.topology[0]]

    def state_eqn(self, current, append=True):
        """This holds the discretized state equations containing the battery dynamics at the pack-level."""
        self.current = current  # added 01/09/22 to fix bug
        dt = self.dt * 3600  # convert from hour to seconds for dynamics equations but not SOC
        self.OCV = np.interp(self.SOC, self.OCV_map_SOC, self.OCV_map_voltage) * self.topology[0]
        self.Ro = (self.B_Ro * np.exp(self.SOC) + self.A_Ro * np.exp(self.C_Ro * self.SOC)) * self.topology[0] / \
                  self.topology[1]
        #   state equations
        # TODO: done: only modify currents for number of cells in parallel
        self.iR1 = np.exp(-dt / (self.R1 * self.C1)) * self.iR1 + (1 - np.exp(-dt / (self.R1 * self.C1))) * current
        self.iR2 = np.exp(-dt / (self.R2 * self.C2)) * self.iR2 + (1 - np.exp(-dt / (self.R2 * self.C2))) * current
        self.voltage = self.OCV + current * self.Ro + self.iR1 * self.R1 + self.iR2 * self.R2
        self.power = self.voltage * self.current / 1000  # kw
        if append:
            self.currents += current,
            self.true_power += self.power,

    def get_OCV(self):
        return np.interp(self.SOC, self.OCV_map_SOC, self.OCV_map_voltage)

    def thermal_dynamics(self):
        """This models the battery's thermal state """
        #   using the lumped-sum capacitance model
        return NotImplementedError


#   TEST THE BATTERY CODE HERE (code below is to sanity-check the battery dynamics)
def test():
    # TODO: include error checking assertion points later
    path_prefix = os.getcwd()
    path_prefix = (path_prefix[: path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation')
    path_prefix.replace('\\', '/')
    battery_config_path = f'{path_prefix}/charging_sim/configs/battery.json'
    with open(battery_config_path, "r") as f:
        battery_config = json.load(f)
    params_list = [key for key in battery_config.keys() if "params_" in key]
    for params_key in params_list:
        print("testing load battery params: ", params_key)
        battery_config[params_key] = np.loadtxt(path_prefix + battery_config[params_key])
    # do the OCV maps as well
    battery_config["OCV_map_voltage"] = np.loadtxt(path_prefix + battery_config["OCV_map_voltage"])[
                                        ::-1]  # ascending order
    battery_config["OCV_map_SOC"] = np.loadtxt(path_prefix + battery_config["OCV_map_SOC"])[::-1]  # ascending order

    buffer_battery = Battery(config=battery_config)
    buffer_battery.battery_setup()
    buffer_battery.load_pack_props()
    buffer_battery.id, buffer_battery.node = 0, 0

    # test dynamics here
    c = -20  # discharging first
    voltages = []
    currents = []
    for _ in range(5):
        buffer_battery.dynamics(c)
        currents.append(c)
    c = 10
    for _ in range(100):
        buffer_battery.dynamics(c)
        currents.append(c)
    c = -10
    for _ in range(2):
        buffer_battery.dynamics(c)
        currents.append(c)
    c = 0
    for _ in range(200):
        buffer_battery.dynamics(c)
        currents.append(c)
    c = 50
    for _ in range(5):
        buffer_battery.dynamics(c)
        currents.append(c)
    c = 0  # charging (Amperes)
    for _ in range(200):
        buffer_battery.dynamics(c)
        currents.append(c)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot_tables(buffer_battery.voltages, label='voltage')
    # ax2.plot(currents, color='k', label='current')
    ax2.plot_tables(buffer_battery.currents, color='r', ls='--', label='adjusted current')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Voltage (V)')
    ax2.set_ylabel('Current (A)')
    ax1.legend()
    plt.legend()
    plt.savefig("battery_test_plot")
    plt.close()
    plt.plot(buffer_battery.SOC_list)
    plt.savefig("SOC_battery_test")
    plt.close()
    plt.plot(buffer_battery.currents)
    plt.xlabel('Time step')
    plt.ylabel('Current (Amperes)')
    print(len(buffer_battery.currents), len(buffer_battery.voltages))
    plt.savefig("currents_battery_test")


if __name__ == "__main__":
    test()
