"""
**Overview**\n
Module for the battery pack class. This module contains the class that loads the battery pack data and structure used
for simulation. The battery pack is composed of individual cells, which are connected in series and parallel to
achieve the desired voltage and capacity.\n

**Example usage**::

 with open(battery_config_path, "r") as f:
    battery_config = json.load(f)
    params_list = [key for key in battery_config.keys() if "params_" in key]

Then the battery can be instantiated as follows::

 for params_key in params_list:
    # This loads the actual battery parameters from the file in which those parameters are stored, prior to
    # instantiating Battery().
    battery_config[params_key] = np.loadtxt(path_prefix + battery_config[params_key])
    # Instantiate the battery using its constructor.
    buffer_battery = Battery(config=battery_config, controller=controller) # controller is the controller for battery
"""

import json
import numpy as np
from utils import num_steps
import matplotlib.pyplot as plt
import os
import pandas as pd


class Battery:
    """
    Each instantiation of this class must include at least a config file, which contains the physical
    constraints and properties of the battery.

    Properties are mainly controlled in the battery config file `battery.json`
        * max-c-rate - determines the power capacity of the cell as a multiple of the value of the energy capacity.
        * max voltage(V) - maximum allowable battery voltage.
        * min Voltage (V) - minimum allowable battery voltage.
        * nominal energy (kWh) - energy deliverable to the battery.
        * id (-).
        * Ambient temperature (Celsius).

    :param battery_type: Type of battery (inconsequential to current dynamics).
    :param node: The node/bus in the distribution network in which the battery resides.
    :param config: Battery configuration file containing the main attributes of the battery.
    :param controller: Controller for the battery.
    :returns: Battery object.
    """

    def __init__(self, battery_type=None, node=None, config=None, controller=None):

        self.node = node  # This is used for battery location.
        self.battery_type = None    #: Initial value: None.
        self.controller = controller    #: Initial value: None.
        self.config = config    #: Initial value: None.

        self.resolution = self.config["resolution"]
        self.dt = self.config["resolution"] / 60  # (Hours)
        self.cap = self.config["cell_nominal_cap"]  # Unit: Ah.
        self.cell_nominal_cap = self.config["cell_nominal_cap"]
        self.R_cell = self.config["resistance"]  # Single cell resistance.
        self.R_pack = None  #: Overall pack resistance.
        self._eff = self.config["round-trip_efficiency"]
        self.max_c_rate = self.config["max_c_rate"]

        self.battery_cost = 200  # this is in $/kWh
        self.To = 273.15 + 23  # Ambient temperature.
        self.T = self.To  # initialize at atmospheric temperature.
        # self.nominal_energy = config["cell_nominal_energy"]     # Watts-hours.

        # LOAD THE PARAMETERS FOR CERTAIN CYCLE INTERVAL
        self.params_0_cycles = self.config["params_0_cycles"]
        self.params_25_cycles = self.config["params_25_cycles"]
        self.params_75_cycles = self.config["params_75_cycles"]
        self.params_125_cycles = self.config["params_125_cycles"]
        # TODO: think about how to load these parameter changes over time
        self.OCV_map_SOC = self.config["OCV_map_SOC"]  # update these two as well
        self.OCV_map_voltage = self.config["OCV_map_voltage"]  # update these two as well

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

        self.max_voltage = self.config["max_cell_voltage"]
        self.pack_max_voltage = self.config["pack_max_voltage"]
        self.min_voltage = self.config["min_cell_voltage"]  # for each cell
        self.pack_max_Ah = self.config["pack_max_Ah"]  # this is used in battery_setup_2 to scale voltage instead of curr
        self.pack_nom_Ah = self.config["pack_nom_Ah"]
        self.cell_nominal_voltage = self.config["nominal_cell_voltage"]  # for each cell
        self.nominal_energy = self.cell_nominal_voltage * self.cell_nominal_cap
        self.Qmax = self.config["SOC_max"] * self.cap
        self.Qmin = self.config["SOC_min"] * self.cap
        self.initial_SOC = self.config["SOC"]
        self.min_SOC = self.config["SOC_min"]
        self.max_SOC = self.config["SOC_max"]
        self.OCV = self.config["max_cell_voltage"]
        self.SOH = self.config["SOH"]
        self.daily_self_discharge = self.config["daily_self_discharge"]

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
        self.start = self.config["start_time"]
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

        self.topology = [1, 1]  # DEFAULT VALUE: singular cell
        self.constraints = None
        self.id = None
        self.location = None
        self.savings = None
        self.state_of_charge = None
        self.nominal_pack_voltage = self.config["pack_voltage"]  # to be initialized later
        self.pack_energy_capacity = self.config["pack_energy_cap"]  # battery rating this is in Watt-hours (Wh)
        self.nominal_pack_cap = None  #: This will be set in :py:meth:battery_setup.
        self.predicted_voltages = []
        self.operating_voltages = []

    def get_true_power(self):
        """Returns the power in or out of the battery.

        :param : None.
        :return : Last power output of battery object."""
        return np.array(self.true_power)

    def battery_setup(self):
        """
        This sets up the series-parallel configuration of the cells,
        given the capacity and voltage of the battery. Scales up Ah capacity, not voltage. Voltage in this setup is
        fixed by the battery json file while the Ah capacity is floating and determined by the given pack voltage and
        Energy Capacity.

        * Energy capacity (Wh).
        * Voltage (V).

        :return: None. Updates battery topology.
        """
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
        Scales up voltage instead of current capacity (Ah), thereby using more cells in series
        for the same battery energy rating. pack_max_AH property is set in config and fixed. Voltage is determined
        by the pack energy capacity (Watt-hours) and the maximum amp-hour capacity.

        * Capacity (Wh).
        * cell_amp_hrs (Ah), cell_voltage (V)

        :param : None.
        :return : None. Updates battery topology.
        """
        # number of modules in parallel should be determined by the power rating and Voltage
        # should use nominal voltage and max allowable current
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

    def est_calendar_aging(self):
        """
        Estimates the constant calendar aging of the battery. this is solely time-dependent.
        Deprecate this later for this object.

        :param: None.
        :return sum(aging_cal): Linear aging result.
        """
        life_cal_years = 10
        seconds_in_min = 60
        seconds_in_year = 31556952
        aging_cal = (self.resolution * seconds_in_min) / (life_cal_years * seconds_in_year)
        aging_cal *= np.ones((num_steps, 1))
        return np.sum(aging_cal)

    def est_cyc_aging(self):
        """Creates linear battery ageing model per hesse. et. Al, and returns its cvx object.
        Deprecate this later for this object.

        :param : None.
        :return cycle aging: CVXPY objective describing cycle aging function.
        """
        seconds_in_min = 60
        life_cyc = 4500  # change this to be input in the config file
        return (0.5 * (np.sum(abs(self.current * self.voltage)) * self.resolution / seconds_in_min)) / \
            (life_cyc / 0.2 * self.nominal_energy)

    def get_power_profile(self, months):
        """
        Returns the power profile of the battery for a certain number of months.

        :param months: Months for which to obtain power profile.
        Returns: Dictionary of power profiles for each month in months.
        """
        return {month: self.power_profile[month] for month in months}

    def get_total_aging(self):
        """Returns the total capacity life loss the battery has experienced so far.

        :param: None.
        :return: Estimated cycle + calendar aging of cell/battery."""
        return self.est_cyc_aging() + self.est_calendar_aging()

    def update_capacity(self):
        """This is not true capacity but anticipated capacity based on linear model.
        This will be deferred to controller. TO BE DEPRECATED.
        """
        aging_value = self.get_aging_value() * 0.2  # do not multiply by 0.2 to keep comparable
        self.total_aging += aging_value
        self.linear_aging += aging_value,
        # self.update_max_current()

    def track_SOC(self, SOC):
        """Updates the states of charge state Vector of the battery.

        :param: SOC - state of charge. This is a self-called function.
        :return: None. """
        self.SOC_track += SOC,
        self.SOC_list += SOC,

    def get_aging_value(self):
        """Returns the actual aging value lost after a cvxpy run.

        :returns: Cycle aging + Calendar aging.
        """
        return self.est_cyc_aging() + self.est_calendar_aging()

    def get_roundtrip_efficiency(self):
        """Returns the estimated round-trip efficiency.

        :returns boolean: self._eff"""
        return self._eff

    def update_max_current(self, verbose=False):
        """Update the max allowable current of the battery - to be deprecated as it is not needed anymore."""
        # self.max_current = self.max_c_rate * self.cap * self.topology[1]
        if verbose:
            print("Maximum allowable current updated.")

    def update_voltage(self, voltage):
        """TO BE DEPRECATED.
        Updates the battery voltage, given the predicted voltage from controller. """
        self.current_voltage = voltage  # I should be updating initial voltage with new voltage measurement
        self.predicted_voltages += voltage,

    def visualize(self, option: str):
        """ TO BE DEPRECATED.
        Plots visualizations and save battery states desired by user.

        :param: option - only uses one option right now."""
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
        """Saves all relevant battery data over the given simulation. Usually called from the battery object
        upon conclusion of simulation.

        :param save_prefix: desired folder in which all the files will be saved.
        :type save_prefix: str.
        :return : None, saves output files in the desired folder.
        """
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

    def dynamics(self, current):
        """Propagates the state of the battery forward one step.
        It takes as input current load (amperes) from the Battery controller and update the battery power.

        :param current: Battery cycle current in amperes.
        :type current: float or np.float().
        :return: Battery response voltage.
        :rtype: Float.
        """
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
        """Updates all properties from cell to pack level using.
        Method used in ref Balogun Et. Al http://dx.doi.org/10.36227/techrxiv.23596725.v2 .
        Uses initial properties/states to update from cell into pack properties for simulation.

        :param: None.
        :return: None."""
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
        """Contains discretized state equations containing the battery dynamics at the pack-level;
        ref here: G. L. Plett, Battery management systems, Volume I: Battery modeling. Artech House, 2015, vol. 1.

        :param boolean append: Decides if to track powers within the ::B battery object.
        :param float current: Current - current in amperes from controller to propagate state forward by one step.
                append - defaults to True. This decides tracking of currents and power over time. Desirable for post-
                optimization analyses.
        :return: None; appends current state into the state history vectors.
        """
        self.current = current  # added 01/09/22 to fix bug
        dt = self.dt * 3600  # convert from hour to seconds for dynamics equations but not SOC
        self.OCV = np.interp(self.SOC, self.OCV_map_SOC, self.OCV_map_voltage) * self.topology[0]
        self.Ro = (self.B_Ro * np.exp(self.SOC) + self.A_Ro * np.exp(self.C_Ro * self.SOC)) * self.topology[0] / \
                  self.topology[1]

        #   state equations
        self.iR1 = np.exp(-dt / (self.R1 * self.C1)) * self.iR1 + (1 - np.exp(-dt / (self.R1 * self.C1))) * current
        self.iR2 = np.exp(-dt / (self.R2 * self.C2)) * self.iR2 + (1 - np.exp(-dt / (self.R2 * self.C2))) * current
        self.voltage = self.OCV + current * self.Ro + self.iR1 * self.R1 + self.iR2 * self.R2
        self.power = self.voltage * self.current / 1000  # kw
        print("Voltage: ", self.voltage)
        if append:
            self.currents += current,
            self.true_power += self.power,

    def get_OCV(self):
        """Uses the Open Circuit Voltage (OCV) Map stored within the battery object to any OCV for any state-of-charge
        SoC via interpolation.
        The OCV-SOC map is obtained a-priori (pre-simulation).

        :param: None.
        :return: Open Circuit Voltage (OCV) at battery's current SoC."""
        return np.interp(self.SOC, self.OCV_map_SOC, self.OCV_map_voltage)

    def thermal_dynamics(self):
        """This models the battery's thermal state. Updates internal and surface temperature of the battery.
        Not available in this version.
        """
        #   using the lumped-sum capacitance model
        # TODO: have someone work on this.
        return NotImplementedError


# #   TEST THE BATTERY CODE HERE (code below is to sanity-check the battery dynamics)
# def test():
#     """This is used to test the battery class module to ensure desired behavior is respected.
#     Saves relevant plots for visual inspection at the end of the simulated test"""
#     # TODO: include error checking assertion points later
#     path_prefix = os.getcwd()
#     path_prefix = (path_prefix[: path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation')
#     path_prefix.replace('\\', '/')
#     battery_config_path = f'{path_prefix}/charging_sim/configs/battery.json'
#     with open(battery_config_path, "r") as f:
#         battery_config = json.load(f)
#     params_list = [key for key in battery_config.keys() if "params_" in key]
#     for params_key in params_list:
#         print("testing load battery params: ", params_key)
#         battery_config[params_key] = np.loadtxt(path_prefix + battery_config[params_key])
#     # do the OCV maps as well
#     battery_config["OCV_map_voltage"] = np.loadtxt(path_prefix + battery_config["OCV_map_voltage"])[
#                                         ::-1]  # ascending order
#     battery_config["OCV_map_SOC"] = np.loadtxt(path_prefix + battery_config["OCV_map_SOC"])[::-1]  # ascending order
#
#     buffer_battery = Battery(config=battery_config)
#     buffer_battery.battery_setup()
#     buffer_battery.load_pack_props()
#     buffer_battery.id, buffer_battery.node = 0, 0
#
#     # test dynamics here
#     c = -20  # discharging first
#     voltages = []
#     currents = []
#     for _ in range(5):
#         buffer_battery.dynamics(c)
#         currents.append(c)
#     c = 10
#     for _ in range(100):
#         buffer_battery.dynamics(c)
#         currents.append(c)
#     c = -10
#     for _ in range(2):
#         buffer_battery.dynamics(c)
#         currents.append(c)
#     c = 0
#     for _ in range(200):
#         buffer_battery.dynamics(c)
#         currents.append(c)
#     c = 50
#     for _ in range(5):
#         buffer_battery.dynamics(c)
#         currents.append(c)
#     c = 0  # charging (Amperes)
#     for _ in range(200):
#         buffer_battery.dynamics(c)
#         currents.append(c)
#
#     fig, ax1 = plt.subplots()
#     ax2 = ax1.twinx()
#     ax1.plot_tables(buffer_battery.voltages, label='voltage')
#     # ax2.plot(currents, color='k', label='current')
#     ax2.plot_tables(buffer_battery.currents, color='r', ls='--', label='adjusted current')
#     ax1.set_xlabel('Time step')
#     ax1.set_ylabel('Voltage (V)')
#     ax2.set_ylabel('Current (A)')
#     ax1.legend()
#     plt.legend()
#     plt.savefig("battery_test_plot")
#     plt.close()
#     plt.plot(buffer_battery.SOC_list)
#     plt.savefig("SOC_battery_test")
#     plt.close()
#     plt.plot(buffer_battery.currents)
#     plt.xlabel('Time step')
#     plt.ylabel('Current (Amperes)')
#     print(len(buffer_battery.currents), len(buffer_battery.voltages))
#     plt.savefig("currents_battery_test")
#
#
# if __name__ == "__main__":
#     test()
