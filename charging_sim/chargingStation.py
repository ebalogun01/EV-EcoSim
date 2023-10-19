"""
Hosts the Charging Station class.
"""

from utils import num_steps
import numpy as np
import matplotlib.pyplot as plt


class ChargingStation:
    """The charging station class produces a load with a power factor parameter that determines its reactive
    load contribution, if any. It also retains all information of all power injection at its grid node/bus.
    It is initialized with its location, capacity, etc. This class ingests the battery, solar and controller modules
    to which it is assigned.

    :param object storage: Storage object assigned to the charging station.
    :param dict config:
    :param object controller: Charging station controller object.
    :param object solar: Charging station solar object. Default is None.
    :param str status: Charging station status. Default 'idle'.
    """
    def __init__(self, storage, config, controller, solar=None, status='idle'):
        self.config = config
        self.id = self.config["locator_index"]
        self.loc = config["location"]
        self.storage = storage
        self.capacity = config["L2_power_cap"] or config["dcfc_power_cap"]
        self.solar = solar
        self.power_factor = config["power_factor"]
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
        """
        Checks if the unit is charging.

        :return: Boolean value indicating if the unit is charging.
        """
        return self.power > self.auxiliary_power

    def update_load(self, net_grid_load, ev_load):
        """
        Updates the charging station loads, including DER assets. MPC mode.

        :param net_grid_load: Net load charging station pulls from the grid.
        :param ev_load: Electric Vehicle charging demand.
        """
        self.current_load = net_grid_load + self.auxiliary_power
        self.loads += net_grid_load,  # net load station pulls from grid, not load from EV
        self.total_load += ev_load + self.auxiliary_power,
        self.solar_power_ev += self.solar.ev_power.value[0, 0],
        self.solar_power_grid += self.solar.grid_power.value[0, 0],
        self.solar_power_battery += self.solar.battery_power.value[0, 0],
        self.pge_blocks += self.controller.pge_gamma.value[0],
        self.storage.predicted_SOC += self.controller.battery_SOC.value[1, 0],  # initial soc is never predicted
        self.storage.pred_power += self.controller.battery_power.value[0, 0],

    def update_load_oneshot(self, net_grid_load, ev_load):
        """
        Updates the charging station loads, including DER assets. Offline mode (Non-MPC).

        :param net_grid_load: Net load charging station pulls from the grid.
        :param ev_load: Electric Vehicle charging demand.
        """
        self.current_load = net_grid_load + self.auxiliary_power
        self.loads.extend(net_grid_load.flatten().tolist())  # net load station pulls from grid, not load from EV
        self.total_load.extend((ev_load + self.auxiliary_power).flatten().tolist())
        self.solar_power_ev.extend(self.solar.ev_power.value.flatten().tolist())
        self.solar_power_grid.extend(self.solar.grid_power.value.flatten().tolist())
        self.solar_power_battery.extend(self.solar.battery_power.value.flatten().tolist())
        self.pge_blocks.extend(self.controller.pge_gamma.value.flatten().tolist())
        self.storage.predicted_SOC.extend(self.controller.battery_SOC.value.flatten().tolist()[1:]) # shape is 1 bigger than others
        self.storage.pred_power.extend(self.controller.battery_power.value.flatten().tolist())

    def is_EV_arrived(self):
        """
        Checks if an EV has arrived at the charging station.

        :return: Boolean value indicating if an EV has arrived at the charging station.
        """
        if self.current_load > 0:
            print("EV is currently at Station ", self.id)
            return True
        return False

    def update_status(self):
        """
        Updates the current status of the EV charging station.

        :return: None.
        """
        if round(self.power[0], 2) > 0:
            self.status = 'in-use'
            print("Charging station is currently occupied.")
        else:
            self.status = 'idle'
            print("Charging station is currently idle.")

    def set_current_load(self, load):
        """
        Sets the current load of the charging station.

        :param load: Load in kW.
        :return: None.
        """
        self.current_load = min(load, self.capacity)

    def get_current_load(self):
        """
        Returns the current load of the charging station.

        :return: Current load (kW) of the charging station.
        """
        return self.current_load

    def save_sim_data(self, save_prefix: str):
        """
        Saves all relevant simulation data to csv files.

        :param save_prefix: Path string to save the data from simulation.
        :return: None.
        """
        import pandas as pd
        save_file_base = f'{str(self.id)}_{self.loc}'
        data = {'Control_current': self.controller.actions,
                'battery_voltage': self.storage.voltages,
                'station_net_grid_load_kW': self.loads,
                'station_total_load_kW': self.total_load,
                'station_solar_load_ev': self.solar_power_ev,
                'station_solar_grid': self.solar_power_grid,
                'station_solar_battery': self.solar_power_battery,
                'battery_power': self.storage.true_power,
                'average_cost_per_interval': self.controller.costs
                }
        if len(self.pge_blocks) > 2:
            print(len(self.pge_blocks))
            data['PGE_power_blocks'] = self.pge_blocks
        elif len(self.pge_blocks) >= 1:
            np.savetxt(f'{save_prefix}/PGE_block_charging_station_sim_{save_file_base}.csv', self.pge_blocks)
        pd.DataFrame(data).to_csv(f'{save_prefix}/charging_station_sim_{save_file_base}.csv')
        print('***** Successfully saved simulation outputs to: ', f'charging_station_sim_{save_file_base}.csv')

    def visualize(self, option=None):
        """
        Visualizing charging station states.

        :param option: plotting option.
        :return: None.
        """
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
