"""This file contains the controller classes used for DER optimization and control within EV-Ecosim"""
import os
import matplotlib.pyplot as plt
from optimization import Optimization
from utils import build_objective, build_electricity_cost, num_steps, build_cost_PGE_BEV2S
import numpy as np
import cvxpy as cp

# Battery_state should include: Estimate SOH corrected from previous day, SOC,

path_prefix = os.getcwd()
path_prefix = (path_prefix[: path_prefix.index('EV50_cosimulation')] + 'EV50_cosimulation')
path_prefix.replace('\\', '/')
OCV_SOC_linear_params = 0


class MPC:
    """This class uses an MPC control scheme for producing control currents to the BESS."""

    def __init__(self, config, storage=None, solar=None):
        self.config = config
        self.resolution = config["resolution"]  # should match object interval? not necessary
        # self.charge_history = np.genfromtxt(path_prefix + config["load_history"]) * 1
        # self.current_testdata = np.genfromtxt(path_prefix + config["simulation_load"])[:-1, ] * 1  # this is used to predict the load, in the future, we will generate a bunch of loads to do this
        # self.reshaped_data = np.reshape(self.current_testdata,
        #                                 self.current_testdata.size)  # flatten data for efficient indexing
        self.storage = storage
        self.storage_constraints = None

        self.load = []
        self.time = 0
        self.w = 0
        self.costs = [0]
        self.control_battery = self.config["control_battery"]
        self.solar = solar

        # SOLAR VARS RELATED TO BATTERY
        if self.solar:
            self.battery_current_solar = cp.Variable((num_steps, 1), nonneg=True)

        # BATTERY VARIABLES (TODO: INCLUDE MIN-MAX SOC AND DISCHARGE REQUIREMENTS DIRECTLY IN CONTROLLER)
        if storage:
            self.battery_ocv_params = None  # to be initialized later
            self.load_battery_ocv()  # loads the battery ocv params above
            self.battery_initial_SOC = self.storage.initial_SOC  # begin with initial information of batt SOC
            self.battery_OCV = self.battery_ocv_params[0][0, 0] * self.battery_initial_SOC + \
                               self.battery_ocv_params[1][0]
            self.battery_capacity = self.storage.cell_nominal_cap
            # controller should be estimating this from time to time. Or decide how it is updated?

        if self.config["electricity_rate_plan"] == "PGEBEV2S":
            self.pge_gamma = cp.Variable(1, integer=True)
            self.pge_gamma_constraint = [self.pge_gamma >= 1]
        self.battery_power = cp.Variable((num_steps, 1))
        self.battery_current_grid = cp.Variable((num_steps, 1), nonneg=True)
        self.battery_current = cp.Variable((num_steps, 1))
        self.battery_current_ev = cp.Variable((num_steps, 1), nonneg=True)
        self.battery_power_ev = cp.Variable((num_steps, 1))
        self.battery_power_grid = cp.Variable((num_steps, 1))

        self.batt_binary_var_ev = cp.Variable((num_steps, 1), boolean=True)
        self.batt_binary_var_grid = cp.Variable((num_steps, 1), boolean=True)
        self.batt_binary_var_solar = cp.Variable((num_steps, 1), boolean=True)
        self.battery_SOC = cp.Variable((num_steps + 1, 1))  # State of Charge max:1 min:0
        self.action = 0
        self.actions = [self.action]

    def initialize_forecast_data(self):
        """loads history to be used for forecasting EV charging"""
        self.charge_history = np.genfromtxt(path_prefix + self.config["load_history"]) * 1
        self.current_testdata = np.genfromtxt(path_prefix + self.config["simulation_load"])[:-1, ] * 1

    def load_battery_ocv(self):
        """Learns the battery OCV Parameters from data and sets the relevant class attribute.
        Inputs - None.
        Returns - None."""
        from sklearn.linear_model import LinearRegression
        indices = ((self.storage.OCV_map_SOC <= 0.9) & (self.storage.OCV_map_SOC >= 0.2)).nonzero()[0]
        soc = self.storage.OCV_map_SOC[indices[0]: indices[-1]].reshape(-1, 1)
        voltage = self.storage.OCV_map_voltage[indices[0]: indices[-1]].reshape(-1, 1)
        model = LinearRegression().fit(soc, voltage)
        self.battery_ocv_params = (model.coef_, model.intercept_)
        # plt.plot(model.coef_ * soc + model.intercept_)
        # plt.savefig("Param_plot.png")
        # plt.close('all')

    def compute_control(self, load, price_vector):
        """
        Optimization-based control actions are computed and passed to the battery.
        Inputs: load - This is the power demand from the charging station.
                price_vector - This is usually the time of use (TOU) rate of the charging station.
        Returns: control_action - Current signals to control the DER system for arbitrage.
        """
        self.time += 1
        control_action = None
        if self.control_battery:
            objective_mode = "Electricity Cost"  # Need to update objective modes to include cost function design
            linear_aging_cost = 0  # based on simple model and predicted control actions - Change this to zero
            electricity_cost = build_cost_PGE_BEV2S(self, load, price_vector, penalize_max_power=False)
            objective = build_objective(objective_mode, electricity_cost, linear_aging_cost)
            opt_problem = Optimization(objective_mode, objective, self, load, self.resolution, None, self.storage,
                                       solar=self.solar, time=0, name="Test_Case_" + str(self.storage.id),
                                       solver=self.config['opt_solver'])
            cost = opt_problem.run()
            self.costs += cost/num_steps,
            if opt_problem.problem.status != 'optimal':
                print('Unable to service travel')
                raise IOError(f'{opt_problem.problem.status}. Suboptimal solution obtained')
            control_action = self.battery_current.value[0, 0]  # this is current flowing through each cell
            self.actions += control_action,
            self.storage.update_capacity()  # to track linear estimated aging
            self.storage.control_current += control_action,  # TODO: double-check this here
        #  need to get all the states here after the first action is taken
        return control_action

    def get_battery_constraints(self, ev_load):
        """
        Creates and updates the battery constraints required to be satisfied by the controller.
        Inputs: ev_load - This is the power demand from the charging station.
        Returns: storage_constraints - List of battery constraints to be respected.
        """
        # TODO: can toggle between battery initial soc and planned soc trajectory
        eps = 1e-8
        cells_series = self.storage.topology[0]
        mod_parallel = self.storage.topology[1]  # parallel modules count
        self.battery_OCV = self.storage.get_OCV() * cells_series  # sensing directly from the battery at each time-step
        self.storage_constraints = \
            [self.battery_SOC[0] == self.battery_initial_SOC,  # changing to deterministic
             self.battery_SOC[1:] == self.battery_SOC[:-1] + (
                     self.resolution / 60 * self.battery_current) / (self.storage.cap * mod_parallel),
             cp.abs(self.battery_current) <= self.storage.max_current,
             self.solar.battery_power == self.battery_current_solar * self.battery_OCV / 1000,
             self.battery_power == self.battery_power_ev + self.battery_power_grid + self.solar.battery_power,
             self.battery_SOC >= self.storage.min_SOC,
             self.battery_SOC <= self.storage.max_SOC,
             self.battery_power_ev == -self.battery_current_ev * self.battery_OCV / 1000,
             self.battery_power_grid == self.battery_current_grid * self.battery_OCV / 1000,
             self.battery_current == self.battery_current_grid + self.battery_current_solar - self.battery_current_ev,

             # included power losses in here
             self.batt_binary_var_ev + self.batt_binary_var_solar <= 1,
             self.batt_binary_var_ev + self.batt_binary_var_grid <= 1,

             self.battery_current_grid <= cp.multiply(self.storage.max_current, self.batt_binary_var_grid),  # GRID
             self.battery_current_solar <= cp.multiply(self.storage.max_current, self.batt_binary_var_solar),  # SOLAR

             self.battery_current_ev <= cp.multiply(self.storage.max_current, self.batt_binary_var_ev),

             # # need to make sure battery is not discharging and charging at the same time with lower 2 constraints
             ev_load + self.battery_power_ev - self.solar.ev_power >= eps  # energy balance
             # allows injecting back to the grid; we can decide if it is wasted or not
             ]
        if self.config["electricity_rate_plan"] == "PGEBEV2S":
            self.storage_constraints.extend(self.pge_gamma_constraint)
        return self.storage_constraints

    def reset_load(self):
        """This is done after one full day is done."""
        self.load = []


class Oneshot:
    """This class uses an offline control scheme for producing control currents to the BESS.
    This is non-MPC and cannot use state feedback in control simulation."""

    def __init__(self, config, storage=None, solar=None, num_steps=96):
        self.config = config
        self.resolution = config["resolution"]  # should match object interval? not necessary
        # self.charge_history = np.genfromtxt(path_prefix + config["load_history"]) * 1
        # self.current_testdata = np.genfromtxt(path_prefix + config["simulation_load"])[:-1, ] * 1  # this is used to predict the load, in the future, we will generate a bunch of loads to do this
        # self.reshaped_data = np.reshape(self.current_testdata,
        #                                 self.current_testdata.size)  # flatten data for efficient indexing
        self.storage = storage
        self.storage_constraints = None
        self.load = []
        self.time = 0
        self.w = 0
        self.costs = [0]
        self.control_battery = self.config["control_battery"]
        self.solar = solar
        self.num_steps = num_steps

        # SOLAR VARS RELATED TO BATTERY
        if self.solar:
            self.battery_current_solar = cp.Variable((num_steps, 1), nonneg=True)

        # BATTERY VARIABLES (TODO: INCLUDE MIN-MAX SOC AND DISCHARGE REQUIREMENTS DIRECTLY IN CONTROLLER)
        if storage:
            self.battery_ocv_params = None  # to be initialized later
            self.load_battery_ocv()  # loads the battery ocv params above
            self.battery_initial_SOC = self.storage.initial_SOC  # begin with initial information of batt SOC
            self.battery_OCV = self.battery_ocv_params[0][0, 0] * self.battery_initial_SOC + \
                               self.battery_ocv_params[1][0]
            self.battery_capacity = self.storage.cell_nominal_cap
            self.battery_eff = self.storage.get_roundtrip_efficiency()
            # controller should be estimating this from time to time. Or decide how it is updated?

        if self.config["electricity_rate_plan"] == "PGEBEV2S":
            self.pge_gamma = cp.Variable(1, integer=True)
            self.pge_gamma_constraint = [self.pge_gamma >= 1]
        self.battery_power = cp.Variable((num_steps, 1))
        self.battery_current_grid = cp.Variable((num_steps, 1), nonneg=True)
        self.battery_current = cp.Variable((num_steps, 1))
        self.battery_current_ev = cp.Variable((num_steps, 1), nonneg=True)
        self.battery_power_ev = cp.Variable((num_steps, 1))
        self.battery_power_grid = cp.Variable((num_steps, 1))

        self.batt_binary_var_ev = cp.Variable((num_steps, 1), boolean=True)
        self.batt_binary_var_grid = cp.Variable((num_steps, 1), boolean=True)
        self.batt_binary_var_solar = cp.Variable((num_steps, 1), boolean=True)
        self.battery_SOC = cp.Variable((num_steps + 1, 1))  # State of Charge max:1 min:0
        self.action = 0
        self.actions = [self.action]

    def load_battery_ocv(self):
        """Learns the battery OCV Parameters from data and sets the relevant class attribute.
        Inputs - None.
        Returns - None."""
        from sklearn.linear_model import LinearRegression
        indices = ((self.storage.OCV_map_SOC <= 0.9) & (self.storage.OCV_map_SOC >= 0.2)).nonzero()[0]
        soc = self.storage.OCV_map_SOC[indices[0]: indices[-1]].reshape(-1, 1)
        voltage = self.storage.OCV_map_voltage[indices[0]: indices[-1]].reshape(-1, 1)
        model = LinearRegression().fit(soc, voltage)
        self.battery_ocv_params = (model.coef_, model.intercept_)
        plt.plot(model.coef_ * soc + model.intercept_)
        plt.savefig("Param_plot.png")
        plt.close('all')

    def compute_control(self, load, price_vector):
        """
        Optimization-based control actions are computed and passed to the battery.
        Inputs: load - This is the power demand from the charging station.
                price_vector - This is usually the time of use (TOU) rate of the charging station.
        Returns: control_action - Current signals to control the DER system for arbitrage.
        """
        control_action = None
        if self.control_battery:
            objective_mode = "Electricity Cost"  # Need to update objective modes to include cost function design
            linear_aging_cost = 0  # based on simple model and predicted control actions - Change this to zero
            electricity_cost = build_cost_PGE_BEV2S(self, load, price_vector, penalize_max_power=False)
            objective = build_objective(objective_mode, electricity_cost, linear_aging_cost)
            opt_problem = Optimization(objective_mode, objective, self, load, self.resolution, None, self.storage,
                                       solar=self.solar, time=0, name=f"Test_Case_{str(self.storage.id)}", solver=self.config['opt_solver'])
            cost = opt_problem.run()
            self.costs.extend([cost/self.num_steps for _ in range(self.num_steps)])
            if opt_problem.problem.status != 'optimal':
                print('Unable to service travel')
                raise Warning("Solution is not optimal!")
            control_action = self.battery_current.value  # this is current flowing through each cell
            self.actions = np.append(np.array(self.actions), control_action)
            self.storage.update_capacity()  # to track linear estimated aging
            self.storage.control_current = control_action  # TODO: double-check this here
        #  need to get all the states here after the first action is taken
        return control_action

    def get_battery_constraints(self, ev_load):
        """ Creates and updates the battery constraints required to be satisfied by the controller.
        Inputs: ev_load - This is the power demand from the charging station.
        Returns: storage_constraints - List of battery constraints to be respected.
        """
        eps = 0.000
        cells_series = self.storage.topology[0]
        mod_parallel = self.storage.topology[1]  # parallel modules count
        self.battery_OCV = self.storage.get_OCV() * cells_series    # sensing directly from the battery at each time-step
        self.storage_constraints = \
            [self.battery_SOC[0] == self.battery_initial_SOC,       # changing to deterministic
             self.battery_SOC[1:] == self.battery_SOC[:-1] + (
                         self.resolution / 60 * self.battery_current) / (self.storage.cap*mod_parallel),
             cp.abs(self.battery_current) <= self.storage.max_current,
             self.solar.battery_power == self.battery_current_solar * self.battery_OCV / 1000,
             self.battery_power == self.battery_power_ev + self.battery_power_grid + self.solar.battery_power,
             # self.battery_power == (self.battery_current * self.battery_OCV) / 1000,
             self.battery_SOC >= self.storage.min_SOC,
             self.battery_SOC <= self.storage.max_SOC,
             self.battery_power_ev == -self.battery_current_ev * self.battery_OCV / 1000,
             self.battery_power_grid == self.battery_current_grid * self.battery_OCV / 1000,
             self.battery_current == self.battery_current_grid + self.battery_current_solar - self.battery_current_ev,

             # included power losses in here
             self.batt_binary_var_ev + self.batt_binary_var_solar <= 1,
             self.batt_binary_var_ev + self.batt_binary_var_grid <= 1,

             self.battery_current_grid <= cp.multiply(self.storage.max_current, self.batt_binary_var_grid),  # GRID
             self.battery_current_solar <= cp.multiply(self.storage.max_current, self.batt_binary_var_solar),  # SOLAR

             self.battery_current_ev <= cp.multiply(self.storage.max_current, self.batt_binary_var_ev),

             # # need to make sure battery is not discharging and charging at the same time with lower 2 constraints
             ev_load + self.battery_power_ev - self.solar.ev_power >= eps  # energy balance
             # allows injecting back to the grid; we can decide if it is wasted or not
             ]
        if self.config["electricity_rate_plan"] == "PGEBEV2S":
            self.storage_constraints.extend(self.pge_gamma_constraint)
        return self.storage_constraints

    def reset_load(self):
        """This is done after one full day is done."""
        self.load = []


class BucketModel:
    """
    This class controls the battery system with the simple bucket model. This is used in comparison with a more dynamic
    model.
    """

    def __init__(self, config, storage=None, solar=None, num_steps=96):
        self.config = config
        self.resolution = config["resolution"]  # should match object interval? not necessary
        self.storage = storage
        self.storage_constraints = None
        self.load = []
        self.time = 0
        self.w = 0
        self.costs = [0]
        self.control_battery = self.config["control_battery"]
        self.solar = solar
        self.num_steps = num_steps

        # SOLAR VARS RELATED TO BATTERY
        # if self.solar:
        #     self.battery_current_solar = cp.Variable((num_steps, 1), nonneg=True)

        # BATTERY VARIABLES (TODO: INCLUDE MIN-MAX SOC AND DISCHARGE REQUIREMENTS DIRECTLY IN CONTROLLER)
        if storage:
            self.battery_ocv_params = None  # to be initialized later
            # self.load_battery_ocv()  # loads the battery ocv params above
            self.battery_initial_SOC = self.storage.initial_SOC  # begin with initial information of batt SOC
            # self.battery_OCV = self.battery_ocv_params[0][0, 0] * self.battery_initial_SOC + \
            #                    self.battery_ocv_params[1][0]
            self.battery_capacity = self.storage.cell_nominal_cap
            self.battery_eff = self.storage.get_roundtrip_efficiency()
            self.battery_energy_capacity = self.storage.pack_energy_capacity

        if self.config["electricity_rate_plan"] == "PGEBEV2S":
            self.pge_gamma = cp.Variable(1, integer=True)
            self.pge_gamma_constraint = [self.pge_gamma >= 1]
        self.battery_power = cp.Variable((num_steps, 1))
        self.battery_power_ev = cp.Variable((num_steps, 1), nonneg=True)
        self.battery_power_grid = cp.Variable((num_steps, 1), nonneg=True)

        self.batt_binary_var_ev = cp.Variable((num_steps, 1), boolean=True)
        self.batt_binary_var_grid = cp.Variable((num_steps, 1), boolean=True)
        self.batt_binary_var_solar = cp.Variable((num_steps, 1), boolean=True)
        self.battery_SOC = cp.Variable((num_steps + 1, 1))  # State of Charge max:1 min:0
        self.action = 0
        self.actions = [self.action]

    def load_battery_ocv(self):
        """Learns the battery OCV Parameters from data and sets the relevant class attribute.
        Inputs - None.
        Returns - None."""
        from sklearn.linear_model import LinearRegression
        indices = ((self.storage.OCV_map_SOC <= 0.9) & (self.storage.OCV_map_SOC >= 0.2)).nonzero()[0]
        soc = self.storage.OCV_map_SOC[indices[0]: indices[-1]].reshape(-1, 1)
        voltage = self.storage.OCV_map_voltage[indices[0]: indices[-1]].reshape(-1, 1)
        model = LinearRegression().fit(soc, voltage)
        self.battery_ocv_params = (model.coef_, model.intercept_)
        plt.plot(model.coef_ * soc + model.intercept_)
        plt.savefig("Param_plot.png")
        plt.close('all')

    def compute_control(self, load, price_vector):
        """
        Optimization-based control actions are computed and passed to the battery.
        Inputs: load - This is the power demand from the charging station.
                price_vector - This is usually the time of use (TOU) rate of the charging station.
        Returns: control_action - Current signals to control the DER system for arbitrage.
        """
        control_action = None
        if self.control_battery:
            objective_mode = "Electricity Cost"  # Need to update objective modes to include cost function design
            linear_aging_cost = 0  # based on simple model and predicted control actions - Change this to zero
            # Todo: Create options for custom electricity cost functions. Design a few others.
            electricity_cost = build_cost_PGE_BEV2S(self, load, price_vector, penalize_max_power=False)
            objective = build_objective(objective_mode, electricity_cost, linear_aging_cost)
            opt_problem = Optimization(objective_mode, objective, self, load, self.resolution, None, self.storage,
                                       solar=self.solar, time=0, name=f"Test_Case_{str(self.storage.id)}",
                                       solver=self.config['opt_solver'])
            cost = opt_problem.run()
            self.costs.extend([cost / self.num_steps for _ in range(self.num_steps)])
            if opt_problem.problem.status != 'optimal':
                print('Unable to service travel')
                raise Warning("Solution is not optimal!")
            control_action = self.battery_power.value  # this is current flowing through each cell
            self.actions = np.append(np.array(self.actions), control_action)
            self.storage.update_capacity()  # to track linear estimated aging
            # self.storage.control_current = control_action  # TODO: double-check this here
        #  need to get all the states here after the first action is taken
        return control_action

    def get_battery_constraints(self, ev_load):
        """
        Creates and updates the battery constraints required to be satisfied by the controller.
        Inputs: ev_load - This is the power demand from the charging station.
        Returns: storage_constraints - List of battery constraints to be respected.
        """
        eps = 0.000
        self.storage_constraints = \
            [self.battery_SOC[0] == self.battery_initial_SOC,  # changing to deterministic
             self.battery_SOC[1:] == self.battery_SOC[:-1] + (
                     self.resolution / 60 * self.battery_power) / (self.battery_energy_capacity/1000),
             cp.abs(self.battery_power) <= self.storage.max_power/1000,
             self.battery_power == -self.battery_power_ev + self.battery_power_grid + self.solar.battery_power,
             self.battery_SOC >= self.storage.min_SOC,
             self.battery_SOC <= self.storage.max_SOC,

             # Binary variables that constrain the decisions of the controller. For example, because the battery cannot
             # charge and discharge at the same time, the sum of the binary variables for charging and discharging
             # should be less than or equal to 1. Ensuring this vital physical constraint is not violated.
             self.batt_binary_var_ev + self.batt_binary_var_solar <= 1,
             self.batt_binary_var_ev + self.batt_binary_var_grid <= 1,  # Battery does not discharge to the grid.

             self.battery_power_grid <= cp.multiply(self.storage.max_power, self.batt_binary_var_grid)/1000,  # GRID
             self.battery_power_ev <= cp.multiply(self.storage.max_power, self.batt_binary_var_ev) / 1000,  # EV
             self.solar.battery_power <= cp.multiply(self.storage.max_power, self.batt_binary_var_solar) / 1000,  # SOLAR

             ev_load - self.battery_power_ev - self.solar.ev_power >= eps  # energy balance, minus here because of sign differences
             # allows injecting back to the grid; we can decide if it is wasted or not
             ]
        if self.config["electricity_rate_plan"] == "PGEBEV2S":
            self.storage_constraints.extend(self.pge_gamma_constraint)
        return self.storage_constraints

    def reset_load(self):
        """This is done after one full day is done."""
        self.load = []