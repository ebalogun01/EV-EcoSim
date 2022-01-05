from config import num_steps
import numpy as np
import json

class ChargingStation:
    """Include the auxiliary power the charging station consumes, add resolution to config as well..."""
    def __init__(self, storage, loc, cap, config, status='idle'):
        self.config = config
        self.id = self.config["locator_index"]
        self.storage = storage
        if self.storage:
            self.storage.id = self.id
        self.loc = loc
        self.capacity = cap
        self.status = status
        self.loads = []
        self.power = np.zeros((num_steps, 1))
        self.current_load = 0
        self.auxiliary_power = 10 # this is in watts
        self.cooling_pump = {}  # properties of the charging station cooling pump
        self.controller = None

    def is_charging(self):
        return self.power > self.auxiliary_power

    def update_load(self, load):
        self.current_load = load
        self.loads.append(load)

    def is_EV_arrived(self):
        if self.current_load > 0:
            print("EV is currently at Station ", self.id)
            return True

    def update_status(self):
        if round(self.power[0], 2) > 0:
            self.status = 'in-use'
            print("Charging station is currently occupied.")
        else:
            self.status = 'idle'
            print("Charging station is currently idle.")

    def set_current_load(self, load):
        self.current_load = load

    def get_current_load(self):
        return self.current_load

    def update_cooling_power(self):
        """Need to define cooling system to vary environmental temps to see how much cooling is needed to maintain
        temperature."""