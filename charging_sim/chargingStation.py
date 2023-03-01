from utils import num_steps
import numpy as np
import matplotlib.pyplot as plt


class ChargingStation:
    """Include the auxiliary power the charging station consumes, add resolution to config as well..."""
    def __init__(self, storage, config, controller, solar=None, status='idle'):
        self.config = config
        self.id = self.config["locator_index"]
        self.loc = config["location"]
        self.storage = storage
        self.capacity = config["L2_power_cap"]     # multiple chargers at a node
        self.capacity_dcfc = config["dcfc_power_cap"]
        self.solar = solar
        self.status = status
        self.loads = [0]
        self.total_load = [0]
        self.solar_power_ev = [0]
        self.solar_power_grid = [0]
        self.solar_power_battery = [0]
        self.power = np.zeros((num_steps, 1))
        self.auxiliary_power = 0.01  # this is in kilo-watts
        self.current_load = self.auxiliary_power
        self.cooling_pump = {}  # properties of the charging station cooling pump
        # COOLING LOAD SHOULD BE A FUNCTION OF CURRENT
        self.controller = controller
        self.pge_blocks = [0]  # this is used with the new pge rate schedule

    def is_charging(self):
        return self.power > self.auxiliary_power

    def update_load(self, net_grid_load, ev_load):
        self.current_load = net_grid_load + self.auxiliary_power
        self.loads += net_grid_load,  # net load station pulls from grid, not load from EV
        self.total_load += ev_load + self.auxiliary_power,
        self.solar_power_ev += self.solar.ev_power.value[0, 0],
        self.solar_power_grid += self.solar.grid_power.value[0, 0],
        self.solar_power_battery += self.solar.battery_power.value[0, 0],
        self.pge_blocks += self.controller.pge_gamma.value[0],

    def update_load_oneshot(self, net_grid_load, ev_load):
        self.current_load += net_grid_load + self.auxiliary_power
        self.loads.extend(net_grid_load.flatten().tolist())  # net load station pulls from grid, not load from EV
        self.total_load.extend((ev_load + self.auxiliary_power).flatten().tolist())
        self.solar_power_ev.extend(self.solar.ev_power.value.flatten().tolist())
        self.solar_power_grid.extend(self.solar.grid_power.value.flatten().tolist())
        self.solar_power_battery.extend(self.solar.battery_power.value.flatten().tolist())
        self.pge_blocks.extend(self.controller.pge_gamma.value.flatten().tolist())

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
        self.current_load = min(load, self.capacity)

    def get_current_load(self):
        return self.current_load

    def update_cooling_power(self):
        """Need to define cooling system to vary environmental temps to see how much cooling is needed to maintain
        temperature."""

    def save_sim_data(self, save_prefix):
        import pandas as pd
        save_file_base = f'{str(self.id)}_{self.loc}'
        data = {'Control_current': [c * self.storage.topology[1] for c in self.controller.actions],
                'battery_voltage': [v * self.storage.topology[0] for v in self.storage.voltages],
                'station_net_grid_load_kW': self.loads,
                'station_total_load_kW': self.total_load,
                'station_solar_load_ev': self.solar_power_ev,
                'station_solar_grid': self.solar_power_grid,
                'station_solar_battery': self.solar_power_battery,
                'battery_power': self.storage.true_power,
                'average_cost_per_interval': self.controller.costs
                }
        if len(self.pge_blocks) > 2:
            data['PGE_power_blocks'] = self.pge_blocks
        pd.DataFrame(data).to_csv(f'{save_prefix}/charging_station_sim_{save_file_base}.csv')
        print('***** Successfully saved simulation outputs to: ', f'charging_station_sim_{save_file_base}.csv')

    def visualize(self, option=None):
        plt.plot(self.controller.actions)
        plt.ylabel("Control current")
        plt.xlabel("Time Steps (count)")
        plt.savefig(f"Control_action_station{self.id}")
        plt.close()
        if option and isinstance(option, str) and option != "storage":
            try:
                data = getattr(self, option)
                plt.plot(data)
                plt.ylabel(option)
                plt.xlabel("Time Steps (count)")
                plt.savefig("Load_profile_station{}".format(self.id))
                plt.close()
            except IOError as e:
                raise IOError("Option chosen is not an attribute! Please choose relevant option") from e
        elif option == "storage":
            battery = getattr(self, option)
            plt.figure()
            plt.plot(battery.voltages, "k", ls="--")
            plt.plot(battery.predicted_voltages)  # currently needs to be fixed, minor bug
            plt.ylabel('Voltage (V)')
            plt.legend(['True Voltage', 'Controller Estimated Voltage'])
            plt.savefig('voltage_plot_{}_{}_Sim.png'.format(battery.id, self.id))
            plt.close()
        else:
            return
