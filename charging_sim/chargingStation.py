from utils import num_steps
import numpy as np
import json
import matplotlib.pyplot as plt

class ChargingStation:
    """Include the auxiliary power the charging station consumes, add resolution to config as well..."""
    def __init__(self, storage, config, controller, status='idle'):
        self.config = config
        self.id = self.config["locator_index"]
        self.storage = storage
        if self.storage:
            self.storage.id = self.id
        self.loc = config["location"]
        self.capacity = config["power_cap"]
        self.status = status
        self.loads = []
        self.power = np.zeros((num_steps, 1))
        self.auxiliary_power = 10 # this is in watts
        self.current_load = self.auxiliary_power
        self.cooling_pump = {}  # properties of the charging station cooling pump
        # COOLING LOAD SHOULD BE A FUNCTION OF CURRENT
        self.controller = controller

    def is_charging(self):
        return self.power > self.auxiliary_power

    def update_load(self, load):
        self.current_load = load + self.auxiliary_power
        self.loads.append(load)     # net load station pulls from grid, not load from EV

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

    def visualize(self, option=None):
        plt.plot(self.controller.actions)
        plt.ylabel("Control current")
        plt.xlabel("Time Steps (count)")
        plt.savefig("Control_action_station{}".format(self.id))
        plt.close()
        if option and isinstance(option, str) and option != "storage":
            try:
                data = getattr(self, option)
                plt.plot(data)
                plt.ylabel(option)
                plt.xlabel("Time Steps (count)")
                plt.savefig("Load_profile_station{}".format(self.id))
                plt.close()
            except:
                raise Exception("Option chosen is not an attribute! Please choose relevant option")
        elif option == "storage":
            battery = getattr(self, option)
            plt.figure()
            plt.plot(battery.true_voltage, "k", ls="--")
            plt.plot(battery.predicted_voltages)
            plt.ylabel('Voltage (V)')
            plt.legend(['True Voltage', 'Controller Estimated Voltage'])
            plt.savefig('voltage_plot_{}_{}_Sim.png'.format(battery.id, self.id))
            plt.close()
        else:
            return

