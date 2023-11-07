import numpy as np
import json


class Node:
    """
    Node object class for defining relevant node_name for co-simulation. It helps organize centralized DER assets within
    their respective grid locations
    """
    def __init__(self, name, storage, solar, controller=None, others=None, children=None, parent=None):
        self.name = name
        self.storage = storage
        self.solar = solar
        self.controller = controller
        if children:
            self.children = children
        else:
            self.children = []

        self.loads = [0]
        self.total_load = [0]
        self.solar_power_ev = [0]
        self.solar_power_grid = [0]
        self.load_data = [0]  # todo: This should be loaded based on input load data
        self.solar_power_storage = [0]
        self.auxiliary_power = 0.00  # this is in kilo-watts
        self.current_load = self.auxiliary_power
        self.cooling_pump = {}  # properties of the charging station cooling pump
        # COOLING LOAD SHOULD BE A FUNCTION OF CURRENT
        self.controller = storage.controller
        self.pge_blocks = [0]  # this is used with the new pge rate schedule

    def update_load(self, net_grid_load, ev_load):
        """
        Updates the node_name loads, including DER assets. MPC mode.

        :param net_grid_load: Net load charging station pulls from the grid.
        :param ev_load: Sum of all electric vehicle charging demand within that primary feeder node_name.
        """
        self.current_load = net_grid_load + self.auxiliary_power
        self.loads += net_grid_load,  # net load station pulls from grid, not load from EV
        self.total_load += ev_load + self.auxiliary_power,
        self.solar_power_ev += self.solar.ev_power.value[0, 0],
        self.solar_power_grid += self.solar.grid_power.value[0, 0],
        self.solar_power_storage += self.solar.battery_power.value[0, 0],
        self.pge_blocks += self.controller.pge_gamma.value[0],
        self.storage.predicted_SOC += self.controller.battery_SOC.value[1, 0],  # initial soc is never predicted
        self.storage.pred_power += self.controller.battery_power.value[0, 0],

    def set_current_load(self, load):
        """
        Sets the current load of the charging station.

        :param load: Load in kW.
        :return: None.
        """
        self.current_load = load

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
        # Save JSON
        with open(f'{save_prefix}/config_{self.id}_{self.loc}.json', "w") as outfile:
            json.dump(self.config, outfile, indent=1)
        print('***** Successfully saved simulation outputs to: ', f'charging_station_sim_{save_file_base}.csv')

    def add_child(self, node: object):
        """
        Adds a child node_name to the current node_name.

        :param node: Node object.
        :return: None.
        """
        self.children.append(node)

