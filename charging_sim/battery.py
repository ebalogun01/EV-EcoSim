import cvxpy as cp
import numpy as np
from config import start_time, num_steps, solar_gen
print('initial imports done')

seconds_in_year = 31556952
seconds_in_min = 60
resolution = 15
OCV_SOC_linear_params = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/OCV_SOC_linear_params_NMC_25degC.npy')
print('ocv done')
# Need to understand charging dynamics as well. Cannot assume symmetry


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
    def __init__(self, battery_type, Q_initial,  node=None):
        batteries = {"Tesla Model 3": {"energy_nom": 0.01746, "SOH": 1, "energy_usable": 0, "round-trip_efficiency": 0.95,
                                       "daily_self-discharge": 1 - 0.0002 / 96,
                                       "SOC_min": 0.4, "SOC_max": 0.99, "life_cal": 13*5, "life_FEC": 4500*5,
                                       "max_c_rate": 4.85/1000, "resistance": 1},
                     "Nissan Leaf": {"energy_nom": 65, "round-trip_efficiency": 0.00, "daily_self-discharge": 0.0000,
                                     "SOC_min": 0.00, "SOC_max": 0.00, "life_cal": 15, "life_FEC": 0000,
                                     "max_c_rate": 20.6}}
        self.cap = 4.85
        self._node = node  # This is used for battery location.
        self.properties = batteries[battery_type]
        self.properties["energy_usable"] = self.properties["energy_nom"] * (self.properties["SOC_max"] -
                                                                            self.properties["SOC_min"])
        self._name = battery_type
        self.topology = None # battery_setup updates with tuple (no_cells_series, no_modules_parallel, total_cells)
        self.ECM_params = [0.1771537, -0.10072846, 0.65168455]
        self.A_Ro = self.ECM_params[0]
        self.B_Ro = self.ECM_params[1]
        self.C_Ro = self.ECM_params[2]

        self._eff = self.properties["round-trip_efficiency"]
        self.SOC_track = []
        self.state_of_charge = None
        self.control_power = np.array([])
        self.OCV = cp.Variable((num_steps, 1))
        self.discharge_current = cp.Variable((num_steps, 1))  # current
        self.discharge_voltage = cp.Variable((num_steps, 1))
        self.true_voltage = np.empty((num_steps+1, 1))  # discharge/charge voltage per time-step
        self.voltage = 3.5
        self.max_voltage = 4.2
        self.min_volatage = 2.5
        self.nominal_voltage = 3.6
        self.Qmax = self.properties["SOC_max"] * self.cap
        self.Qmin = self.properties["SOC_min"] * self.cap
        self.Q = cp.Variable((num_steps + 1, 1))  # Amount of energy Kwh available in battery
        self.SOC = cp.Variable((num_steps + 1, 1))  # State of Charge max:1 min:0
        self.SOH = self.properties["SOH"]
        self.Q_initial = Q_initial
        self.control_current = np.array([])

        self.power_charge = cp.Variable((num_steps, 1))
        self.power_discharge = cp.Variable((num_steps, 1))
        self.current = cp.Variable((num_steps, 1))
        self.true_power = np.array([])
        self.start = start_time
        self.MPC_Control = {'Q': [], 'P': []}  # tracks the control actions for the MPC control
        self.size = 100
        self.cell_count = 0
        self.id = None # Use Charging Station ID for this
        self.savings = None
        self.total_aging = 0
        self.true_capacity_loss = 0
        self.resistance_growth = 0
        self.true_SOC = np.empty((num_steps+1, 1)) # BatterySim updates this with the true SOC different from Optimization
        self.track_true_aging = []  # want to keep track to observe trends based on degradation models
        self.track_linear_aging = []  # linear model per Hesse Et. Al
        self.constraints = None
        self.ambient_temp = 25  # Celsius
        self.charging_costs = 0
        self.power_profile = {'Jan': [], 'Feb': [], 'Mar': [], 'Apr': [], 'May': [], 'Jun': [], 'Jul': [], 'Aug': [],
                              'Sep': [], 'Oct': [], 'Nov': [], 'Dec': []}
        self.Ro = self.get_Ro()

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
        no_cells_series = voltage//cell_voltage
        no_modules_parallel = capacity//cell_amp_hrs
        total_cells = no_cells_series * no_modules_parallel
        self.topology = (no_cells_series, no_modules_parallel, total_cells)
        print("Battery initialized.\n",
              "Capacity is {} Ah".format(capacity),
              "Total number of cells is: {} .".format(total_cells))
        C1 = "null"
        C2 = "null"
        R1 = "null"
        R2 = "null"
        return total_cells

    def est_calendar_aging(self):
        """Estimates the constant calendar aging of the battery. this is solely time-dependent."""
        life_cal_years = self.properties["life_cal"]
        aging_cal = (resolution * seconds_in_min) / (life_cal_years * seconds_in_year)
        aging_cal *= np.ones((num_steps, 1))
        return np.sum(aging_cal)

    def est_cyc_aging(self):
        """Creates linear battery ageing model, and returns its cvx object."""
        life_cyc = self.properties["life_FEC"]
        nominal_energy = self.properties["energy_nom"]  # kWh
        aging_cyc = (0.5 * (cp.sum(self.power_charge + self.power_discharge)
                            * resolution / seconds_in_min)) / (life_cyc/0.2 * nominal_energy)
        return aging_cyc

    def get_power_profile(self, month):
        return self.power_profile[month]

    def get_total_aging(self):
        return self.est_cyc_aging() + self.est_calendar_aging()

    def get_aging_value(self):
        """returns the actual aging value lost after a cvxpy run..."""
        aging_value = self.est_cyc_aging().value + self.est_calendar_aging()
        return aging_value

    def update_capacity(self):
        """This is not true capacity but anticipated capacity based on linear model."""
        aging_value = self.get_aging_value() * 0.2  # do not multiply by 0.2 to keep comparable
        # self.properties["SOH"] -= aging_value
        # self.SOH = self.properties["SOH"]
        # self.Qmax = self.properties["SOC_max"] * self.properties["energy_nom"] * self.SOH
        self.total_aging += aging_value
        self.track_linear_aging.append(aging_value)  # this is to ADD aging for each time step

    def get_Ro(self):
        """This can be updated later to include current..."""
        Ro = self.B_Ro * cp.exp(self.SOC) + self.A_Ro * cp.exp(self.C_Ro * self.SOC) + self.resistance_growth
        return Ro

    def update_SOC(self):
        action_length = 1
        self.Q_initial = self.Q.value[action_length]

    def track_SOC(self, SOC):
        pass

    def get_properties(self):
        return self.properties

    def learn_params(self):
        pass

    def get_constraints(self, EV_load):
        # print(EV_load)
        print("ev load shape", EV_load.shape)
        print(self.start, self.start+num_steps)
        print("solar shape is", solar_gen[self.start:self.start + num_steps].shape)
        eps = 1e-6 # This is a numerical artifact. Values tend to solve at very low negative values but this helps avoid it.
        # num_cells_series = self.topology[0]
        # num_modules_parallel = self.topology[1]
        num_cells = self.topology[2]
        """Need to convexify constraints"""
        self.constraints = [self.Q[0] == self.Q_initial,
                            self.OCV == OCV_SOC_linear_params[0] * self.SOC[0:num_steps] + OCV_SOC_linear_params[1],
                            self.discharge_voltage == self.OCV + cp.multiply(self.current, 0.076),  # approx from t=1 not 0.
                            self.Q[1:num_steps + 1] == resolution / 60 * self.current + self.Q[0:num_steps],
                            # self.power_charge - self.power_discharge == cp.multiply(
                            #     self.current, self.OCV) - cp.multiply(self.current**2, 0.076),
                            self.power_charge - self.power_discharge == cp.multiply(self.nominal_voltage, self.current)/1000, # kw
                            self.Q >= self.cap/3,
                            self.Q <= self.cap,
                            self.SOC == self.Q / 4.85,
                            self.power_charge >= 0,
                            self.power_discharge >= 0,
                            # cp.abs(self.current) <= self.properties["max_c_rate"],
                            # self.power_discharge <= self.properties["max_c_rate"],
                            EV_load + num_cells * (self.power_charge - self.power_discharge) -
                            solar_gen[self.start:self.start + num_steps] >= eps
                            # no injecting back to the grid; should try unconstrained. This could be infeasible.
                            ]
        return self.constraints

