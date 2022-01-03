from optimization import Optimization
from config import energy_prices_TOU, build_objective, build_electricity_cost, \
    resolution, num_steps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)
# Battery_state should include: Estimate SOH corrected from previous day, SOC,

# Get the load predictive models
LSTM1 = tf.keras.models.load_model("/home/ec2-user/EV50_cosimulation/DLMODELS/LSTM_01.h5")
LSTM2 = tf.keras.models.load_model("/home/ec2-user/EV50_cosimulation/DLMODELS/LSTM_ONE_STEP.h5")

"""Build different MPC classes for different horizons maybe? It's hard to do one MPC for all horizons due to ML-side"""


class MPC:

    def __init__(self, past_data, current_testdata, scaled_onestep_data, storage, std_test, mean_test):
        self.past_data = past_data
        self.current_testdata = current_testdata  # this is used to predict the load
        self.reshaped_data = np.reshape(current_testdata, current_testdata.size)
        self.one_step_data = self.reshaped_data[-48:]  # one step LSTM uses last 48 time steps to predict the next
        self.std_test = std_test
        self.mean_test = mean_test
        self.storage = storage
        self.load = np.array([])
        self.scaled_onestep_data = scaled_onestep_data[0]
        self.scaler_onestep = scaled_onestep_data[1]
        self.full_day_prediction = np.array([])
        self.resolution = 15    # minutes
        # self.LSTM_model = tf.keras.models.load_model("LSTM_01.h5")

    def compute_control(self, start, shift, stop, battery_size):
        predicted_load = np.reshape(self.predict_load(start, shift, stop), (96, 1))
        # print(predicted_load)
        battery_constraints = self.storage.get_constraints(predicted_load)  # battery constraints
        objective_mode = "All"  # Need to update objective modes to include cost function design
        linear_aging_cost = self.storage.get_total_aging()  # based on simple model and predicted control actions
        electricity_cost = build_electricity_cost(self.storage, predicted_load)  # based on prediction as well
        # transformer_cost not included here for now
        objective = build_objective(objective_mode, electricity_cost, linear_aging_cost)
        opt_problem = Optimization(objective_mode, objective, battery_constraints, predicted_load, resolution, None,
                                   self.storage, time=0, name="Test_Case_" + str(self.storage.id))
        opt_problem.run()

        if opt_problem.problem.status != 'optimal':
            print('Unable to service travel')
        if electricity_cost.value < 0:
            print('Negative Electricity Cost')
            print(electricity_cost.value, self.storage.current.value)
        control_actions = np.hstack([self.storage.power_charge.value[0], self.storage.power_discharge.value[0]])

        # only take first action
        self.storage.Q_initial = self.storage.Q.value[1]  # estimated new SOC or Battery Capacity left
        print(self.storage.Q_initial)
        self.storage.SOC_track.append(self.storage.Q.value[1] / self.storage.Qmax)
        self.storage.update_capacity()  # to track linear estimated aging
        # obtain the true state of charge from the batteryAgingSim (How frequently though?)
        if len(self.storage.control_current) < num_steps: # UPDATING ALL POWER TO CURRENTS
            self.storage.control_current = np.append(self.storage.control_current, self.storage.current.value[0])
        else:
            self.storage.control_current = self.storage.current.value[0] # it should be only updating one but then
            # I am showing results after each step I think. Need to also fix the compute current scheme currently being used

        #  need to get all the states here after the first action is taken
        return control_actions, predicted_load

    def predict_load(self, start, shift, stop, days_length=14):
        """this uses two ML models for predictions. One for full day prediction (runs only once a day) and the
        other for time-step update"""
        begin = stop * 96 - 48 + shift  # shift at each time-step, then reset after a day is done
        end = begin + 48
        test_input_onestep = np.reshape(self.scaled_onestep_data[begin:end], (1, 48, 1))
        if not self.full_day_prediction.any():   # This checks if a full day is done
            test_input_fullday = np.reshape(self.reshaped_data[start:start + days_length * num_steps],
                                            (1, days_length, num_steps))
            # print(test_input_fullday.shape)
            self.full_day_prediction = LSTM1.predict(test_input_fullday) * self.std_test + self.mean_test
            self.full_day_prediction.shape = (num_steps, 1)
        prediction_next_step = self.scaler_onestep.inverse_transform(LSTM2.predict(test_input_onestep))
        index = len(self.load) + 1  # this is tracking what time step we are at
        prediction = np.append(self.load, prediction_next_step)
        prediction = np.append(prediction, self.full_day_prediction[index:, :])
        # print(prediction.shape)
        return prediction

    def reset_load(self):
        """This is done after one full day is done."""
        self.load = []
        self.full_day_prediction = np.array([])

class MPC2:
    """this uses a different prediction and control mechanism..to be developed later"""
    pass

