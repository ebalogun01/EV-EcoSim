from config import num_steps
print('numsteps')
import numpy as np


class ChargingStation:
    """Include the auxiliary power the charging station consumes"""
    def __init__(self, storage, loc, cap, ID, status='idle'):
        self.id = ID
        self.storage = storage
        self.loc = loc
        self.capacity = cap
        self.status = status
        self.power = np.zeros((num_steps, 1))
        self.auxiliary_power = 10 # this is in watts
        self.cooling_pump = {}  # properties of the charging station cooling pump

    def is_charging(self):
        return self.power > self.auxiliary_power

    def is_EV_arrived(self):
        if self.power[0] > 0:
            print("EV is currenttly at Station ", self.id)
            return True

    def update_status(self):
        if round(self.power[0], 2) > 0:
            self.status = 'in-use'
            print("Charging station is currently occupied.")
        else:
            self.status = 'idle'
            print("Charging station is currently idle.")

    def update_cooling_power(self):
        """Need to define cooling system to vary environmental temps to see how much cooling is needed to maintain
        temperature."""