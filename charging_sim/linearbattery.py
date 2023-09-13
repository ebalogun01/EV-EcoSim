"""
This module contains the program for the battery pack class.

**Usage**\n
Proper usage is done by the :module:orchestrator.py module.


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
    buffer_battery = Battery(config=battery_config, controller=controller)

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
        self.battery_type = None  #: Initial value: None.
        self.controller = controller  #: Initial value: None.
        self.config = config  #: Initial value: None.

        self.resolution = self.config["resolution"]
        self.dt = self.config["resolution"] / 60  # (Hours)
        self.cap = self.config["cell_nominal_cap"]  # Unit: Ah.
        self.cell_nominal_cap = self.config["cell_nominal_cap"]
        self.R_cell = self.config["resistance"]  # Single cell resistance.
        self.R_pack = None  #: Overall pack resistance.
        self._eff = self.config["round-trip_efficiency"]
        self.max_c_rate = self.config["max_c_rate"]

        # LOAD THE PARAMETERS FOR CERTAIN CYCLE INTERVAL.
        self.params_0_cycles = self.config["params_0_cycles"]
        self.params_25_cycles = self.config["params_25_cycles"]
        self.params_75_cycles = self.config["params_75_cycles"]
        self.params_125_cycles = self.config["params_125_cycles"]
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
        self.pack_max_Ah = self.config[
            "pack_max_Ah"]  # this is used in battery_setup_2 to scale voltage instead of curr
        self.pack_nom_Ah = self.config["pack_nom_Ah"]
        self.cell_nominal_voltage = self.config["nominal_cell_voltage"]  # for each cell
        self.nominal_energy = self.cell_nominal_voltage * self.cell_nominal_cap
        self.nominal_pack_voltage = self.config["pack_voltage"]  # to be initialized later
        self.pack_energy_capacity = self.config["pack_energy_cap"]  # battery rating this is in Watt-hours (Wh)
        self.max_power = self.max_c_rate * self.pack_energy_capacity
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
        self.total_kWh_thruput = 0.0
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

        :return: None. Updates battery topology."""
        # number of modules in parallel should be determined by the power rating and Voltage
        # should use nominal voltage and max allowable current
        print("**** Pre-initialized nominal pack voltage is {}".format(self.nominal_pack_voltage))
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
        print("Est. tot. no. of cycles is: ", self.total_kWh_thruput / ((self.cell_nominal_cap + self.cap) / 2),
              'cycles')

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
        self.min_voltage = self.min_voltage * self.topology[0]
        self.max_current *= self.topology[1]
        self.voltages = [self.voltage * self.topology[0]]

    def save_sim_data(self, save_prefix):
        """Saves all relevant battery data over the given simulation. Usually called from the battery object
        upon conclusion of simulation.

        :param save_prefix: desired folder in which all the files will be saved.
        :type save_prefix: str.
        :return : None, saves output files in the desired folder.
        """
        save_file_base = f'{str(self.id)}_{self.node}'
        # print(len(self.predicted_SOC), len(self.SOC_track), len(self.SOH_track), np.array(self.voltages).shape,
        #         np.array(self.currents).shape, np.array(self.cycle_aging).shape, np.array(self.calendar_aging).shape,
        #         np.array(self.true_power).shape, np.array(self.pred_power).shape)
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
        total_cycles = 0.5 * self.total_kWh_thruput*1000 / self.pack_energy_capacity
        np.savetxt(f'{save_prefix}/total_batt_cycles_{save_file_base}.csv', [total_cycles])
        np.savetxt(f'{save_prefix}/pack_resistance_{save_file_base}.csv', [self.R_pack])
        print('***** Successfully saved simulation outputs to: ', f'battery_sim_{save_file_base}.csv')
        print("Est. tot. no. of cycles is: ", total_cycles, 'cycles')

    # def dynamics(self, power):
    #     """Propagates the state of the battery forward one step.
    #     Uses simple linear bucket battery model.
    #     It takes as input power (kW) from the Battery controller and update the battery power.
    #
    #     :param power: Battery cycle current in amperes.
    #     :type power: float or np.float().
    #     :return: None.
    #     """
    #     self.power = power  # Convert to kW
    #     self.voltage = self.get_OCV() * self.topology[0]
    #     self.SOC = self.SOC + power * self.dt / (self.pack_energy_capacity/1000)
    #     self.current = self.power*1000 / self.voltage   # This is in amperes.
    #     self.voltage -= self.current * self.R_pack
    #     if self.voltage < self.min_voltage:
    #         self.voltage = self.min_voltage
    #     elif self.voltage > self.max_voltage:
    #         self.voltage = self.max_voltage
    #     self.current = self.power*1000 / self.voltage    # readjust current.
    #     if self.current > self.max_current:
    #         self.current = self.max_current
    #     elif self.current < -self.max_current:
    #         self.current = -self.max_current
    #
    #     # Now correct power for voltage drop.
    #     self.power = (self.voltage * self.current)/1000     # convert to kW.
    #
    #     self.voltages += self.voltage,
    #     self.true_power += self.power,
    #     self.currents += self.current,
    #     self.track_SOC(self.SOC)
    #     self.total_kWh_thruput += abs(self.power) * self.dt  # cycle counting

    def dynamics(self, power):
        """Propagates the state of the battery forward one step.
        Uses simple linear bucket battery model.
        It takes as input power (kW) from the Battery controller and update the battery power.

        :param power: Battery cycle current in amperes.
        :type power: float or np.float().
        :return: None.
        """
        # todo: not consider aging and consider aging.
        self.power = power  # Convert to kW
        self.voltage = self.get_OCV() * self.topology[0]
        self.SOC = self.SOC + power * self.dt / (self.pack_energy_capacity/1000)
        self.current = self.power*1000 / self.voltage   # This is in amperes.
        self.current = self.power*1000 / self.voltage    # readjust current.

        self.voltages += self.voltage,
        self.true_power += self.power,
        self.currents += self.current,
        self.track_SOC(self.SOC)
        self.total_kWh_thruput += abs(self.power) * self.dt  # cycle counting

    def get_OCV(self):
        """Uses the Open Circuit Voltage (OCV) Map stored within the battery object to any OCV for any state-of-charge
        SoC via interpolation.
        The OCV-SOC map is obtained a-priori (pre-simulation).

        :param: None.
        :return: Open Circuit Voltage (OCV) at battery's current SoC."""
        return np.interp(self.SOC, self.OCV_map_SOC, self.OCV_map_voltage)
