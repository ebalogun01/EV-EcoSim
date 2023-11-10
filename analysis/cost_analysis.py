"""
This module contains the CostEstimator Class, which estimates the cost of the different grid and DER components
from the simulation. This is used for the post-simulation cost calculations
"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# Defaults.
PLOT_FONT_SIZE = 16
plt.rcParams.update({'font.size': PLOT_FONT_SIZE})


class CostEstimator:
    """
    This class is used to calculate levelized cost of DER assets in EV-Ecosim.
    The LCOE is the levelized cost of energy, which is defined as the estimated revenue or total net expenditure
    required to build and operate an energy system over a specified cost recovery period. The LCOE normalizes the entire
    system cost by the energy throughput to compare the economics energy devices that would otherwise be challenging to
    compare.

    :param num_days: The number of days for which the calculation is run.
    """

    def __init__(self, num_days):
        """
        Constructor method.

        :param num_days: The number of days for which the calculation is run.
        """
        self.solar_rating = None
        self.trans_price_dict = {}
        self.dcfc = False  # boolean for determining if transformer is 480 or 240V
        self.num_days = num_days
        self.trans_Th = None  # hot-spot temp
        self.trans_To = None  # top-oil temp
        self.TOU_rates = None
        self.battery_cost = None  # to be calculated later
        self.solar_price_per_m2 = 6
        self.battery_price_per_kWh = 345  # ($) source: https://www.nrel.gov/docs/fy21osti/79236.pdf
        self.trans_cost_per_kVA = None  # create a non-linear cost curve for these prices (I sense batteries are the same)
        self.trans_normal_life = 180000  # hours
        self.resolution = 15
        self.TOU_rates = np.loadtxt('../elec_rates/PGE_BEV2_S_annual_TOU_rate_15min.csv')[
                         :96 * self.num_days]  # change this to referenced property
        # todo: cannot find good source for 2400/240V transformer prices

    def calculate_battery_cost(self, result_dir):
        """
        Calculates the battery costs and updates the different cost components, including LCOE.

        :param result_dir: Directory in which to save the results dictionary.
        :return dict result_dict: Dictionary of results.
        """
        current_dir = os.getcwd()
        os.chdir(result_dir)
        with open('scenario.json', "r") as f:
            scenario = json.load(f)
        result_dict = {}
        capital_cost = self.battery_price_per_kWh / 1000 * scenario['battery']['pack_energy_cap']
        for root, dirs, files, in os.walk(".", topdown=True):
            for file in files:
                path_lst = file.split("_")
                if 'battery' in path_lst and 'plot.png' not in path_lst:
                    battery_LOL = 1 - pd.read_csv(file)['SOH'].to_numpy()[-1]
                    avg_daily_energy_thruput = np.abs(pd.read_csv(file)['power_kW'].to_numpy()[1:]).sum() \
                                               * self.resolution / 60 * 1 / self.num_days
                    expected_life_days = 0.2 / (battery_LOL / self.num_days)
                    expected_energy_thruput_over_lifetime = avg_daily_energy_thruput * expected_life_days
                    capital_loss_to_aging = (battery_LOL / 0.2 * capital_cost)
                    self.battery_cost = capital_cost + capital_loss_to_aging
                    result_dict[f'battery_sim_{path_lst[2]}'] = {"capital_loss_aging": capital_loss_to_aging,
                                                                 "capital_loss_aging_per_day": capital_loss_to_aging / expected_life_days,
                                                                 "capital_cost": capital_cost,
                                                                 "battery_LOL": battery_LOL,
                                                                 "LOL_per_day": battery_LOL / self.num_days,
                                                                 "battery_total_cost": self.battery_cost,
                                                                 "total_cost_per_day": self.battery_cost / expected_life_days,
                                                                 'lcoe': self.battery_cost / expected_energy_thruput_over_lifetime,
                                                                 'lcoe_aging': capital_loss_to_aging / expected_energy_thruput_over_lifetime
                                                                 }
        with open("postopt_cost_batt.json", 'w') as config_file_path:
            json.dump(result_dict, config_file_path, indent=1)  # save to JSON
        os.chdir(current_dir)  # go back to initial dir
        return result_dict

    def calculate_cable_cost(self, length, underground=True, voltage="HV", cores=3, core_girth=25):
        """
        Values are pulled from the DACE Price booklet
        Ref: https://www.dacepricebooklet.com/table-costs/high-and-low-voltage-underground-electrical-power-cables-0
        """
        cost_per_m = 28.6  # TODO table lookup
        return length * cost_per_m

    def calculate_transformer_cost(self, capacity):
        """
        Values are pulled from the DACE Price booklet
        Ref: https://www.dacepricebooklet.com/table-costs/standard-transformers-10-kv-400-v-oil-cooled-0

        Oil cooled transformers for indoor installation. Installed and connected. Capacity: 10 kV to 400 V.

        Included:
            conservator;
            gas relay or nitrogen blanket.

        Excluded:
            architectural facilities;
            discounts;
            cable work.

        :param capacity: Capacity in kVA
        :return: Assumed transformer cost in USD."""

        costs = pd.read_csv("analysis/configs/transformer_costs.csv")
        cost = costs.at[0]['Price from']
        for index, row in costs.iterrows():
            if capacity <= row['Capacity']:
                break
            cost = row['Price from']
            print(row['Price from'])
        cost = cost * 1.1  # adjustment from EUR to USD
        return cost

    def calculate_solar_cost(self):
        """
        Values are pulled from the NREL solar cost calculator.
        Ref: https://www.nrel.gov/solar/market-research-analysis/solar-levelized-cost.html
        To be deprecated soon.

        :return: None
        """
        return

    def calculate_electricity_cost_PGEBEV2s(self, result_dir, PGE_separate_file=True):
        """
        Calculates the overall electricity PGEBEV2S cost for a given scenario.

        :param str result_dir: Directory in which the result is saved.
        :param PGE_separate_file:
        :return dict result_dict: A dictionary comprising all the cost components and their dollar amounts.
        """
        current_dir = os.getcwd()
        os.chdir(result_dir)
        result_dict = {}
        price_per_block = 95.56  # ($/Block) # need to make these agnostic for now just leave as is
        overage_fee = 3.82  # ($/kW)
        for root, dirs, files, in os.walk(".", topdown=True):
            for file in files:
                path_lst = file.split("_")
                if 'station' in path_lst and 'block' not in path_lst and 'plot.png' not in path_lst:
                    net_grid_load = pd.read_csv(file)['station_net_grid_load_kW'].to_numpy()[1:]
                    total_grid_load = pd.read_csv(file)['station_total_load_kW'].to_numpy()[1:]
                    net_ev_grid_load_plusbatt = total_grid_load - pd.read_csv(file)['station_solar_load_ev'].to_numpy()[
                                                                  1:] + pd.read_csv(file)['battery_power'].to_numpy()[
                                                                        1:]
                    total_energy = total_grid_load.sum() * self.resolution / 60
                    max_load = total_grid_load.max()
                    average_load = total_grid_load.mean()
                    self.plot_loads(total_grid_load, net_ev_grid_load_plusbatt, prefix=f'{file.split(".")[0]}_',
                                    labels=["Total demand", "Net demand with DER"])
                    if PGE_separate_file:
                        block_subscription = int(np.loadtxt(f'PGE_block_{file}')[1])
                    else:
                        block_subscription = int(pd.read_csv(file)['PGE_power_blocks'].to_numpy()[1])
                    subscription_cost = block_subscription * price_per_block  # This is in blocks of 50kW which makes it very convenient ($/day)
                    penalty_cost = max((np.max(net_grid_load) - 50 * block_subscription), 0) * overage_fee  # ($)
                    TOU_cost = np.sum(
                        self.TOU_rates[0:net_grid_load.shape[0]] * net_grid_load) * self.resolution / 60  # ($)
                    electricity_cost = TOU_cost + penalty_cost + subscription_cost
                    result_dict[f'charging_station_sim_{path_lst[3]}'] = {"TOU_cost": TOU_cost,
                                                                          "subscription_cost": subscription_cost,
                                                                          "penalty_cost": penalty_cost,
                                                                          "total_elec_cost": electricity_cost,
                                                                          "cost_per_day": electricity_cost / self.num_days,
                                                                          "max_load": max_load,
                                                                          "avg_load": average_load,
                                                                          "cost_per_kWh": electricity_cost / total_energy}
                elif 'battery' in path_lst and 'plot.png' not in path_lst:
                    power = pd.read_csv(file)['power_kW'].to_numpy()[1:]
                    power_pred = pd.read_csv(file)['pred_power_kW'].to_numpy()[1:]
                    soc = pd.read_csv(file)['SOC'].to_numpy()[1:]
                    soc_pred = pd.read_csv(file)['SOC_pred'].to_numpy()[1:]
                    self.plot_soc(soc, soc_pred, prefix=f'{file.split(".")[0]}_SOC',
                                  labels=['true soc', 'pred soc', 'SoC'])
                    self.plot_power(power, power_pred, prefix=f'{file.split(".")[0]}_power',
                                    labels=['true power', 'pred power', 'power (kW)'])

        with open("postopt_cost_charging.json", 'w') as config_file_path:
            json.dump(result_dict, config_file_path, indent=1)  # save to JSON
        os.chdir(current_dir)  # go back to initial dir
        return result_dict

    def transformer_cost(self):
        """Cannot find good resource data for this yet."""
        return NotImplementedError

    @staticmethod
    def plot_loads(total_load, net_load, prefix=None, labels: list = None):
        """
        Creates plots overlaying load and net loads for post-simulation visualization.

        :param total_load: Overall EV load demand at node, can include building load if controllable.
        :param net_load: total_load minus DER buffer.
        :param prefix: Plot file label prefix.
        :param labels: Legend labels for each plotted curve.
        :return: None.
        """
        plt.close('all')
        num_days = 1
        start_day = 0
        num_steps = num_days * 96
        fig, ax = plt.subplots()
        x_vals = 15 / 60 * np.arange(0, num_steps)
        total_load_plot = total_load[start_day * 96:start_day * 96 + num_steps]
        net_load_plot = net_load[start_day * 96:start_day * 96 + num_steps]

        lb = np.zeros(num_steps)
        alph = 0.3
        interp = True
        total_load_color = 'blue'
        net_load_color = 'orange'
        ax.plot(x_vals, total_load_plot, color=f'tab:{total_load_color}')
        ax.plot(x_vals, net_load_plot, color=f'tab:{net_load_color}')
        ax.fill_between(x_vals, lb, total_load_plot, color=f'tab:{total_load_color}',
                        label=labels[0], interpolate=interp, alpha=alph)
        ax.fill_between(x_vals, lb, net_load_plot, color=f'tab:{net_load_color}', label=labels[1], interpolate=interp,
                        alpha=alph)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=round(max(x_vals)))
        plt.xlabel('Hour of day')
        plt.ylabel('Power (kW)')
        plt.legend()
        fig.tight_layout()
        if prefix:
            plt.savefig(f'{prefix}_load_plot.png')
            plt.close('all')
            return
        plt.savefig('loads.png')

    @staticmethod
    def plot_soc(soc, soc_pred, prefix=None, labels: list = None):
        """
        Plots the controller predicted and true state of charge of the battery system.

        :param soc: True state of charge.
        :param soc_pred: Controller predicted state of charge.
        :param prefix: Plot file label prefix.
        :param labels: Legend labels for each plotted curve.
        :return: None.
        """
        error_abs_mean = np.mean(np.abs((soc - soc_pred) / (soc + 1e-6)) * 100)
        MAPE = np.max(np.abs((soc - soc_pred) / (soc + 1e-6)) * 100)
        np.savetxt('abs_percent_err_soc.csv', [error_abs_mean])
        np.savetxt('MAPE_soc.csv', [MAPE])
        plt.close('all')
        num_days = 10
        start_day = 0
        num_steps = num_days * 96
        fig, ax = plt.subplots()
        x_vals = 15 / 60 * np.arange(0, num_steps)
        soc_plot = soc[start_day * 96:start_day * 96 + num_steps]
        soc_pred_plot = soc_pred[start_day * 96:start_day * 96 + num_steps]

        ax.plot(x_vals, soc_plot, color='tab:blue', label=labels[0])
        ax.plot(x_vals, soc_pred_plot, '--', color='tab:red', label=labels[1])
        plt.xlabel('Hour of day')
        plt.ylabel(labels[2])
        plt.legend()
        fig.tight_layout()
        if prefix:
            plt.savefig(f'{prefix}_soc_plot.png')
            return
        plt.savefig('soc_plot.png')

    @staticmethod
    def plot_power(power, power_pred, prefix=None, labels: list = None):
        """
        Plots the controller predicted and true power of the battery system.

        :param power: True power.
        :param power_pred: Controller predicted power.
        :param prefix: Plot file label prefix.
        :param labels: Legend labels for each plotted curve.
        :return: None.
        """
        error_abs_mean = np.mean(np.abs((power - power_pred) / (power + 1e-6)) * 100)
        MAPE = np.max(np.abs((power - power_pred) / (power + 1e-6)) * 100)  # Mean Absolute Percent Error.
        np.savetxt('abs_percent_err_power.csv', [error_abs_mean])
        np.savetxt('MAPE_power.csv', [MAPE])
        plt.close('all')
        num_days = 10
        start_day = 0
        num_steps = num_days * 96
        fig, ax = plt.subplots()
        x_vals = 15 / 60 * np.arange(0, num_steps)
        power_plot = power[start_day * 96:start_day * 96 + num_steps]
        power_pred_plot = power_pred[start_day * 96:start_day * 96 + num_steps]

        ax.plot(x_vals, power_plot, color='tab:blue', label=labels[0])
        ax.plot(x_vals, power_pred_plot, '--', color='tab:red', label=labels[1])
        plt.xlabel('Hour of day')
        plt.ylabel(labels[2])
        plt.legend()
        fig.tight_layout()
        if prefix:
            plt.savefig(f'{prefix}_power_plot.png')
            return
        plt.savefig('power_plot.png')

    def solar_cost(self, result_dir):
        """
        Calculates the overall capital cost of the solar system. This will give the dollar cost of the solar system
        used for the charging station design problem.

        Not fully implemented.
        
        :param str result_dir: Location to save the result.
        :return: Solar PV capital cost.
        """
        return self.solar_rating * 1e6 / 150 * self.solar_price_per_m2  # approximate area using 150W/m2

    def calculate_trans_loss_of_life(self, result_dir):
        """
        Estimates the expected transformer loss of life. The transformer loss of life (or LOL) is modelled as a function
        of the hot-spot temperature.

        Reference:

        * 5.11.3 of IEEE Std C57.12.00-2010 a minimum normal insulation life expectancy of 180 000 hours is expected.

        :param result_dir: Directory in which the loss results is saved.
        :return: Dictionary of transformer losses.
        """
        current_dir = os.getcwd()
        os.chdir(result_dir)
        with open('scenario.json', "r") as f:
            scenario = json.load(f)
        # save the charging buses
        result_dict = {}
        for root, dirs, files, in os.walk(".", topdown=True):
            for file in files:
                if 'trans_Th' in file:
                    trans_Th_data = pd.read_csv(file)
        relevant_dcfc_trans = [trans for trans in trans_Th_data.columns if 'dcfc' in trans]
        relevant_L2_trans = [f'trip_trans_{trans.split("_")[-1]}' for trans in scenario['L2_nodes']]
        trans_str_list = relevant_L2_trans + relevant_dcfc_trans
        relevant_Th_data = trans_Th_data[trans_str_list]
        # ref for A and B https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4073181
        A = -27.558
        B = 14573
        for trans in trans_str_list:
            trans_Th = relevant_Th_data[trans]
            F_EQA = np.mean(np.exp(15000 / 383 - 15000 / (trans_Th + 273)))  # equivalent aging factor
            print("F_EQA: ", F_EQA)
            percent_LOL = F_EQA * (24 * self.num_days) * 100 / self.trans_normal_life
            result_dict[trans] = percent_LOL
            result_dict[f'{trans}_LOL_per_day'] = percent_LOL / self.num_days
        result_dict["average_LOL"] = sum(result_dict.values()) / (len(trans_str_list))

        with open("postopt_trans_lol.json", 'w') as config_file_path:
            json.dump(result_dict, config_file_path, indent=1)  # save to JSON
        os.chdir(current_dir)  # go back to initial dir
        return result_dict
