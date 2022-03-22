import cvxpy as cp
import numpy as np
from utils import start_time, num_steps, solar_gen
import matplotlib.pyplot as plt

seconds_in_year = 31556952
seconds_in_min = 60
resolution = 15
OCV_SOC_linear_params = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/OCV_SOC_linear_params_NMC_25degC.npy')

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
        self.dt = config["resolution"] / 60
        self.cap = config["cell_nominal_cap"]  # Ah
        self.nominal_cap = config["cell_nominal_cap"]
        self.cell_resistance = config["resistance"]  # TO be updated
        self._eff = config["round-trip_efficiency"]
        self.ECM_params = config["ECM_params"]["Ro"]
        self.max_c_rate = config["max_c_rate"]
        self.max_current = self.max_c_rate * self.cap
        self.A_Ro = self.ECM_params[0]
        self.B_Ro = self.ECM_params[1]
        self.C_Ro = self.ECM_params[2]
        self._name = battery_type
        self.iR1 = 0
        self.iR2 = 0

        self.voltage = config["cell_nominal_voltage"]
        self.max_voltage = config["max_cell_voltage"]
        self.min_voltage = config["max_cell_voltage"]  # for each cell
        self.nominal_voltage = config["min_cell_voltage"]  # for each cell
        self.nominal_energy = config["cell_nominal_voltage"] * config["cell_nominal_cap"]
        self.Qmax = config["SOC_max"] * self.cap
        self.Qmin = config["SOC_min"] * self.cap
        self.initial_SOC = config["SOC"]
        self.min_SOC = config["SOC_min"]
        self.max_SOC = config["SOC_max"]
        self.SOH = config["SOH"]
        self.daily_self_discharge = config["daily_self_discharge"]

        # battery_setup updates with tuple (no_cells_series, no_modules_parallel, total_cells)

        self.SOC_track = [self.initial_SOC]
        self.SOH_track = [self.SOH] # to be finished later
        self.control_power = np.array([])
        self.OCV = cp.Variable((num_steps, 1))
        self.discharge_current = cp.Variable((num_steps, 1))  # current
        self.discharge_voltage = cp.Variable((num_steps, 1))
        self.voltages = list()  # to be updated at each time-step (seems expensive)
        self.current_voltage = 0
        self.true_voltage = np.array([])  # discharge/charge voltage per time-step
        self.Q = cp.Variable((num_steps + 1, 1))  # Amount of energy Kwh available in battery
        self.SOC = cp.Variable((num_steps + 1, 1))  # State of Charge max:1 min:0

        self.Q_initial = Q_initial  # include the units here
        self.control_current = np.array([])

        self.power_charge = cp.Variable((num_steps, 1))
        self.power_discharge = cp.Variable((num_steps, 1))
        self.power = cp.Variable((num_steps, 1))
        self.current = cp.Variable((num_steps, 1))
        self.true_power = list()
        self.start = start_time
        self.MPC_Control = {'Q': [], 'P': []}  # tracks the control actions for the MPC control
        self.size = 100  # what does this mean?
        self.cell_count = 0

        self.total_aging = 0
        self.true_capacity_loss = 0
        self.resistance_growth = 0
        self.true_SOC = np.empty((num_steps + 1, 1))  #  BatterySim updates this with the true SOC different from Optimization
        self.true_aging = []  # want to keep track to observe trends based on degradation models
        self.linear_aging = []  # linear model per Hesse Et. Al

        self.ambient_temp = 25  # Celsius
        self.charging_costs = 0
        self.power_profile = {'Jan': [], 'Feb': [], 'Mar': [], 'Apr': [], 'May': [], 'Jun': [], 'Jul': [], 'Aug': [],
                              'Sep': [], 'Oct': [], 'Nov': [], 'Dec': []}
        self.Ro = self.get_Ro()

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

    def battery_setup(self, voltage, capacity, cell_params):
        """
        Capacity (Wh)
        Voltage (V)
        params: (cell_amp_hrs (Ah), cell_voltage (V)
        TODO: Finish this up and scale battery up completely.
            Do comparison algo setup with real battery and parameter adjustment."""
        cell_amp_hrs = cell_params[0]
        cell_voltage = cell_params[1]
        no_cells_series = voltage // cell_voltage
        no_modules_parallel = capacity // cell_amp_hrs
        self.cell_count = no_cells_series * no_modules_parallel
        self.topology = (no_cells_series, no_modules_parallel, self.cell_count)
        self.nominal_pack_voltage = no_cells_series * self.nominal_voltage
        self.nominal_pack_cap = no_modules_parallel * self.cap
        print("***** Battery initialized. *****\n",
              "Capacity is {} Ah".format(capacity),
              "Total number of cells is: {} .".format(self.cell_count))
        return self.cell_count

    def est_calendar_aging(self):
        """Estimates the constant calendar aging of the battery. this is solely time-dependent."""
        life_cal_years = 10
        aging_cal = (resolution * seconds_in_min) / (life_cal_years * seconds_in_year)
        aging_cal *= np.ones((num_steps, 1))
        return np.sum(aging_cal)

    def est_cyc_aging(self):
        """Creates linear battery ageing model per hesse. et al, and returns its cvx object."""
        life_cyc = 4500 # change this to be input in the config file
        aging_cyc = (0.5 * (cp.sum(self.power_charge + self.power_discharge)
                            * resolution / seconds_in_min)) / (life_cyc / 0.2 * self.nominal_energy)
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
        aging_value = self.est_cyc_aging().value + self.est_calendar_aging()
        return aging_value

    def update_capacity(self):
        """This is not true capacity but anticipated capacity based on linear model."""
        aging_value = self.get_aging_value() * 0.2  # do not multiply by 0.2 to keep comparable
        self.total_aging += aging_value
        self.linear_aging.append(aging_value)  # this is to ADD aging for each time step

    def get_Ro(self):
        """This can be updated later to include current..."""
        Ro = self.ECM_params[1] * cp.exp(self.SOC) + self.ECM_params[0] * cp.exp(self.C_Ro * self.SOC) \
             + self.resistance_growth
        return Ro

    def update_SOC(self):
        action_length = 1
        self.Q_initial = self.Q.value[action_length]

    def track_SOC(self, SOC):
        self.SOC_track.append(SOC)

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
            if option == "SOC":
                plt.figure()
                plt.plot(self.SOC_track)
                plt.xlabel("Time Step")
                plt.ylabel("State of Charge")
                plt.savefig("SOC_{}.png".format(self.id))
                plt.close()

    def visualize_voltages(self):
        pass

    def learn_params(self):
        pass

    def update_params(self):
        """This changes the battery params depending on the number of cycles"""
        pass
    
    def dynamics(self, current):
        # TODO: define the battery dynamics here

        dt = self.dt * 3600  # convert from hour to seconds
        # state equations

        self.SOC = self.SOC - current * self.dt / self.cap

        if self.SOC >= self.SOC_max:
            print('out of bounds, readjusting voltage...')
            self.SOC = self.SOC_max
            self.SOC_track.append(self.SOC)
            self.voltage = self.max_voltage
            self.voltages = np.append(self.voltages, self.voltage)
            return self.voltage
        if self.SOC <= self.SOC_min:
            print('out of bounds, readjusting voltage...')
            self.voltage = self.min_voltage
            self.SOC = self.SOC_min
            self.SOC_track.append(self.SOC)
            self.voltages = np.append(self.voltages, self.voltage)
            return self.voltage

        self.OCV = np.interp(self.SOC, self.map[:, 0], self.map[:, 1])
        self.Ro = self.B_Ro * np.exp(self.SOC) + self.A_Ro * np.exp(self.C_Ro * self.SOC)

        #   state equations
        self.iR1 = np.exp(-dt / (self.R1 * self.C1)) * self.iR1 + (1 - np.exp(-dt / (self.R1 * self.C1))) * current
        self.iR2 = np.exp(-dt / (self.R2 * self.C2)) * self.iR2 + (1 - np.exp(-dt / (self.R2 * self.C2))) * current
        self.voltage = self.OCV - current * self.Ro - self.iR1 * self.R1 - self.iR2 * self.R2
        if self.voltage > self.max_voltage:
            self.voltage = self.max_voltage
            print('max voltage exceeded')
            self.voltages = np.append(self.voltages, self.voltage)  # numpy array
            self.SOC_track.append(self.SOC)
            return self.voltage
        if self.voltage < self.min_voltage:
            self.voltage = self.min_voltage
            print('min voltage exceeded')
            self.voltages = np.append(self.voltages, self.voltage)  # numpy array
            self.SOC_track.append(self.SOC)
            return self.voltage

        self.voltages = np.append(self.voltages, self.voltage)  # numpy array
        self.SOC_track.append(self.SOC)
        return self.voltage




    # def get_constraints(self, EV_load):
    #     eps = 1e-9  # This is a numerical artifact. Values tend to solve at very low negative values but this helps avoid it.
    #     num_cells_series = self.topology[0]
    #     num_modules_parallel = self.topology[1]
    #     num_cells = self.topology[2]
    #     self.constraints = [self.SOC[0] == self.initial_SOC,
    #                         self.OCV == OCV_SOC_linear_params[0] * self.SOC[0:num_steps] + OCV_SOC_linear_params[1],
    #                         self.discharge_voltage == self.OCV + cp.multiply(self.current, 0.076),
    #                         self.power == cp.multiply(self.nominal_pack_voltage, self.current*num_modules_parallel)/1000,  # kw
    #                         self.power == self.power_charge + self.power_discharge,
    #                         # self.Q >= self.Qmin,  # why did I do this?
    #                         # self.Q <= self.Qmax,
    #                         self.SOC[1:num_steps + 1] == self.SOC[0:num_steps] - (self.daily_self_discharge/num_steps) \
    #                         + (resolution / 60 * self.current)/self.cap,
    #                         self.power_discharge <= 0,
    #                         self.power_charge >= 0,
    #                         self.SOC >= self.min_SOC,
    #                         self.SOC <= self.max_SOC,
    #                         EV_load + (self.power_charge + self.power_discharge) - solar_gen[self.start:self.start + num_steps] >= eps
    #                         # no injecting back to the grid; should try unconstrained. This could be infeasible.
    #                         ]
    #     return self.constraints

    # START THINKING ABOUT THE SEPARATION OF OPTIMIZATION FROM BATTERY
