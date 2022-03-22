from optimization import Optimization
from utils import build_objective, build_electricity_cost, resolution, num_steps, start_time, num_steps, solar_gen
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import cvxpy as cp

# Battery_state should include: Estimate SOH corrected from previous day, SOC,

# Get the load predictive models
LSTM1 = tf.keras.models.load_model("/home/ec2-user/EV50_cosimulation/DLMODELS/LSTM_01.h5")
LSTM2 = tf.keras.models.load_model("/home/ec2-user/EV50_cosimulation/DLMODELS/LSTM_ONE_STEP.h5")
OCV_SOC_linear_params = np.load('/home/ec2-user/EV50_cosimulation/BatteryData/OCV_SOC_linear_params_NMC_25degC.npy')

"""Build different MPC classes for different horizons maybe? It's hard to do one MPC for all horizons due to ML-side"""


class MPC:

    def __init__(self, config, storage):
        self.config = config
        self.resolution = config["resolution"]  # should match object interval? not necessary
        self.charge_history = np.genfromtxt(config["load_history"])
        self.current_testdata = np.genfromtxt(config["simulation_load"])[:-1, ] / 10 # this is used to predict the load
        self.reshaped_data = np.reshape(self.current_testdata, self.current_testdata .size) # flatten data for efficient indexing
        self.one_step_data = self.reshaped_data[-48:]  # one step LSTM uses last 48 time steps to predict the next
        self.std_data = np.std(self.current_testdata, 0)   # from training distribution
        self.std_data[self.std_data == 0] = 1
        self.mean_data = np.mean(self.current_testdata, 0)
        self.scaled_test_data = (self.current_testdata - self.mean_data) / self.std_data    # should use history
        self.storage = storage
        self.storage_constraints = None
        self.load = np.array([])
        self.w = 0

        self.scaler_onestep = MinMaxScaler()
        load_history_onestep = np.reshape(self.charge_history, (self.charge_history.size, 1))  # scaling based on historical dist
        self.scaler_onestep.fit(load_history_onestep)
        # self.scaled_test_data_onestep = [self.scaler_onestep.transform(np.reshape(test_data, (test_data.size, 1))), scaler]

        self.scaled_onestep_data = self.scaler_onestep.transform(
            np.reshape(self.current_testdata, (self.current_testdata.size, 1)))
        # self.scaler_onestep = scaled_onestep_data[1]
        self.full_day_prediction = np.array([])
        self.action = 0
        self.actions = []
        # self.LSTM_model = tf.keras.models.load_model("LSTM_01.h5")

    def initialize_forecast_data(self):
        """loads history to be used for forecasting EV charging"""
        self.charge_history = np.genfromtxt(config["load_history"])
        self.current_testdata = np.genfromtxt(config["simulation_load"])[:-1, ] / 10

    def compute_control(self, start, shift, stop, battery_size, price_vector):
        predicted_load = np.reshape(self.predict_load(start, shift, stop), (96, 1))

        # print(predicted_load)
        battery_constraints = self.get_battery_constraints(predicted_load)  # battery constraints
        objective_mode = "Electricity Cost"  # Need to update objective modes to include cost function design
        linear_aging_cost = self.storage.get_total_aging()  # based on simple model and predicted control actions
        electricity_cost = build_electricity_cost(self.storage, predicted_load, price_vector)  # based on prediction as well
        objective = build_objective(objective_mode, electricity_cost, linear_aging_cost)
        opt_problem = Optimization(objective_mode, objective, battery_constraints, predicted_load, resolution, None,
                                   self.storage, time=0, name="Test_Case_" + str(self.storage.id))
        opt_problem.run()
        # print("Charge ", "Discharge ", self.storage.power_charge.value[0], self.storage.power_discharge.value[0])
        # print("POWER: ", self.storage.power.value)
        if opt_problem.problem.status != 'optimal':
            print('Unable to service travel')
        if electricity_cost.value < 0:
            print('Negative Electricity Cost')
            # print(electricity_cost.value, self.storage.current.value)
        control_actions = np.hstack([self.storage.power_charge.value[0], self.storage.power_discharge.value[0]])
        self.actions.append(self.storage.current.value[0, 0])
        # print(control_actions.shape, "control actions shape check")
        self.storage.update_voltage(self.storage.discharge_voltage.value[0, 0])
        # plt.plot(self.storage.discharge_voltage.value)
        # plt.plot(self.storage.OCV.value)
        # plt.legend(["OCV", "discharge voltage"])
        # plt.savefig("Voltage_simulation_plot_{}".format(self.w))
        # plt.close()
        # self.w += 1
        # only take first action
        # self.storage.Q_initial = self.storage.Q.value[1]  # estimated new SOC or Battery Capacity left
        # self.storage.SOC_track.append(self.storage.Q.value[1] / self.storage.Qmax)
        self.storage.update_capacity()  # to track linear estimated aging
        # obtain the true state of charge from the batteryAgingSim (How frequently though?)
        if len(self.storage.control_current) < num_steps:
            self.storage.control_current = np.append(self.storage.control_current, self.storage.current.value[0])
        else:
            self.storage.control_current = self.storage.current.value[0] # it should be only updating one but then
            # I am showing results after each step I think. Need to also fix the compute current scheme currently being used

        #  need to get all the states here after the first action is taken
        return control_actions, predicted_load

    def predict_load(self, start, shift, stop, days_length=14):
        print("shift", shift)
        """this uses two ML models for predictions. One for full day prediction (runs only once a day) and the
        other for time-step update"""
        begin = stop * 96 - 48 + shift  # shift at each time-step, then reset after a day is done
        end = begin + 48
        test_input_onestep = np.reshape(self.scaled_onestep_data[begin:end], (1, 48, 1))
        if not self.full_day_prediction.any():   # This checks if a full day is done
            test_input_fullday = np.reshape(self.reshaped_data[start:start + days_length * num_steps],
                                            (1, days_length, num_steps))
            # print(test_input_fullday.shape)
            self.full_day_prediction = LSTM1.predict(test_input_fullday) * self.std_data + self.mean_data
            self.full_day_prediction.shape = (num_steps, 1)
        prediction_next_step = self.scaler_onestep.inverse_transform(LSTM2.predict(test_input_onestep))
        index = len(self.load) + 1  # this is tracking what time step we are at
        # print("controller load is: ", self.load)
        prediction = np.append(self.load, prediction_next_step)     # include previous day's known load
        # print("pred", len(prediction))
        prediction = np.append(prediction, self.full_day_prediction[index:, :])
        # print(prediction.shape)
        return prediction

    def get_battery_constraints(self, EV_load):
        eps = 1e-9  # This is a numerical artifact. Values tend to solve at very low negative values but
        # this helps avoid it.
        num_cells_series = self.storage.topology[0]
        num_modules_parallel = self.storage.topology[1]
        num_cells = self.storage.topology[2]
        self.storage_constraints = [self.storage.SOC[0] == self.storage.initial_SOC,
                            self.storage.OCV == OCV_SOC_linear_params[0] * self.storage.SOC[0:num_steps] + OCV_SOC_linear_params[1],
                            self.storage.discharge_voltage == self.storage.OCV + cp.multiply(self.storage.current, 0.076),
                            self.storage.power == cp.multiply(self.storage.nominal_pack_voltage,
                                                      self.storage.current * num_modules_parallel) / 1000,  # kw
                            self.storage.power == self.storage.power_charge + self.storage.power_discharge,
                            self.storage.SOC[1:num_steps + 1] == self.storage.SOC[0:num_steps] - (self.storage.daily_self_discharge / num_steps) \
                            + (resolution / 60 * self.storage.current) / self.storage.cap,
                            self.storage.power_discharge <= 0,
                            self.storage.power_charge >= 0,
                            self.storage.SOC >= self.storage.min_SOC,
                            self.storage.SOC <= self.storage.max_SOC,
                            EV_load + (self.storage.power_charge + self.storage.power_discharge) - solar_gen[
                                                                                   self.storage.start:self.storage.start + num_steps] >= eps
                            # no injecting back to the grid; should try unconstrained. This could be infeasible.
                            ]
        return self.storage_constraints

    def reset_load(self):
        """This is done after one full day is done."""
        self.load = np.array([])
        self.full_day_prediction = np.array([])

class MPC2:
    """this uses a different prediction and control mechanism..to be developed later"""
    pass

