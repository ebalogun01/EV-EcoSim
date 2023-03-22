import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

'''This will be used for post simulation additional charging station cost analyses'''


class CostEstimator:
    def __init__(self, num_days):
        self.trans_price_dict = {}
        self.dcfc = False  # boolean for determining if transformer is 480 or 240V
        self.num_days = num_days
        # self.data = data
        self.trans_Th = None  # hot-spot temp
        self.trans_To = None  # top-oil temp
        self.TOU_rates = None
        self.battery_cost = None  # to be calculated later
        # self.scenario = scenario
        self.solar_price_per_m2 = 6
        # self.solar_rating = scenario["solar_rating"]
        self.battery_price_per_kWh = 345  # ($) source: https://www.nrel.gov/docs/fy21osti/79236.pdf
        self.trans_cost_per_kVA = None  # create a non-linear cost curve for these prices (I sense batteries are the same)
        self.trans_normal_life = 180000  # hours
        self.resolution = 15
        self.TOU_rates = np.loadtxt('../elec_rates/PGE_BEV2_S_annual_TOU_rate_15min.csv')[:96 * self.num_days]
        # todo: cannot find good source for 2400/240V transformer prices

    def calculate_battery_cost(self, result_dir):
        current_dir = os.getcwd()
        os.chdir(result_dir)
        with open('scenario.json', "r") as f:
            scenario = json.load(f)
        result_dict = {}
        capital_cost = self.battery_price_per_kWh / 1000 * scenario['pack_energy_cap']
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
                                                                 'lcoe_aging': capital_loss_to_aging /expected_energy_thruput_over_lifetime
                                                                 }
        with open("postopt_cost_batt.json", 'w') as config_file_path:
            json.dump(result_dict, config_file_path, indent=1)  # save to JSON
        os.chdir(current_dir)  # go back to initial dir
        return result_dict

    def calcultate_solar_cost(self):
        """ref: https://www.nrel.gov/solar/market-research-analysis/solar-levelized-cost.html """
        lcoe = 350
        return lcoe

    def calculate_LCOE(self):
        """Estimate levelized cost of electricity for storage"""

    def calculate_electricity_cost_PGEBEV2s(self, result_dir, PGE_seperate_file=True):
        """Calculates the overall electricity cost for that scenario"""
        current_dir = os.getcwd()
        os.chdir(result_dir)
        result_dict = {}
        price_per_block = 95.56  # ($/Block)
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
                    total_energy = total_grid_load.sum() * self.resolution/60
                    max_load = total_grid_load.max()
                    average_load = total_grid_load.mean()
                    self.plot_loads(total_grid_load, net_ev_grid_load_plusbatt, prefix=f'{file.split(".")[0]}_',
                                    labels=["total demand", "net demand with DER"])
                    if PGE_seperate_file:
                        block_subscription = int(np.loadtxt(f'PGE_block_{file}')[1])
                    else:
                        block_subscription = int(pd.read_csv(file)['PGE_power_blocks'].to_numpy().max())
                    subscription_cost = block_subscription * price_per_block  # This is in blocks of 50kW which makes it very convenient ($/day)
                    penalty_cost = max((np.max(net_grid_load) - 50 * block_subscription), 0) * overage_fee  # ($)
                    TOU_cost = np.sum(self.TOU_rates * net_grid_load) * self.resolution / 60  # ($)
                    electricity_cost = TOU_cost + penalty_cost + subscription_cost
                    result_dict[f'charging_station_sim_{path_lst[3]}'] = {"TOU_cost": TOU_cost,
                                                                          "subscription_cost": subscription_cost,
                                                                          "penalty_cost": penalty_cost,
                                                                          "total_elec_cost": electricity_cost,
                                                                          "cost_per_day": electricity_cost / self.num_days,
                                                                          "max_load": max_load,
                                                                          "avg_load": average_load,
                                                                          "cost_per_kWh": electricity_cost/total_energy}
                elif 'battery' in path_lst and 'plot.png' not in path_lst:
                    soc = pd.read_csv(file)['power_kW'].to_numpy()[1:]
                    soc_pred = pd.read_csv(file)['pred_power_kW'].to_numpy()[1:]
                    self.plot_soc(soc, soc_pred, prefix=f'{file.split(".")[0]}_SOC',
                                  labels=['true power', 'pred power', 'power (kW)'])

        with open("postopt_cost_charging.json", 'w') as config_file_path:
            json.dump(result_dict, config_file_path, indent=1)  # save to JSON
        os.chdir(current_dir)  # go back to initial dir
        return result_dict

    def transformer_cost(self):
        """Cannot find good resource data for this yet"""
        return NotImplementedError

    @staticmethod
    def plot_loads(total_load, net_load, prefix=None, labels: list = None):
        plt.close('all')
        num_days = 1
        start_day = 0
        num_steps = num_days * 96
        fig, ax = plt.subplots()
        x_vals = 15 / 60 * np.arange(0, num_steps)
        total_load_plot = total_load[start_day * 96:start_day * 96 + num_steps]
        net_load_plot = net_load[start_day * 96:start_day * 96 + num_steps]

        lb = np.zeros(num_steps)
        alph = 1
        interp = True
        ax.plot(x_vals, total_load_plot, color='tab:red')
        ax.plot(x_vals, net_load_plot, color='tab:green')
        ax.fill_between(x_vals, lb, total_load_plot, color='tab:red',
                        label=labels[0], interpolate=interp, alpha=alph)

        ax.fill_between(x_vals, lb, net_load_plot, color='tab:green',
                        where=(total_load_plot >= net_load_plot), label=labels[1], interpolate=interp, alpha=alph)

        ax.fill_between(x_vals, total_load_plot, net_load_plot, where=(total_load_plot <= net_load_plot),
                        color='tab:green', interpolate=interp, alpha=alph)
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
        error_abs_mean = np.mean(np.abs((soc - soc_pred) / (soc + 1e-4)) * 100)
        MAPE = np.max(np.abs((soc - soc_pred) / (soc + 1e-4)) * 100)
        np.savetxt('abs_percent_err.csv', [error_abs_mean])
        np.savetxt('MAPE.csv', [MAPE])
        plt.close('all')
        num_days = 10
        start_day = 0
        num_steps = num_days * 96
        fig, ax = plt.subplots()
        x_vals = 15 / 60 * np.arange(0, num_steps)
        soc_plot = soc[start_day * 96:start_day * 96 + num_steps]
        soc_pred_plot = soc_pred[start_day * 96:start_day * 96 + num_steps]

        alph = 0.8
        ax.plot(x_vals, soc_plot, color='tab:blue', label=labels[0])
        ax.plot(x_vals, soc_pred_plot, '--', color='tab:red', label=labels[1])

        # ax.set_ylim(bottom=0)
        # ax.set_xlim(left=0, right=round(max(x_vals)))
        plt.xlabel('Hour of day')
        plt.ylabel(labels[2])
        plt.legend()
        fig.tight_layout()
        if prefix:
            plt.savefig(f'{prefix}_soc_plot.png')
            return
        plt.savefig('soc_plot.png')

    def transformer_degradation_cost(self):
        cost = self.trans_cost_per_kVA
        cost_per_kVA = 0
        operation_cost = 0
        return NotImplementedError

    def solar_cost(self, result_dir):
        return self.solar_rating * 1e6 / 150 * self.solar_price_per_m2  # approximate area using 150W/m2

    def calculate_trans_loss_of_life(self, result_dir):
        """Implement transformer degradation model post-simulation"""
        # Per 5.11.3 of IEEE Std C57.12.00-2010, a minimum normal insulation life expectancy of 180 000 hours is expected
        current_dir = os.getcwd()
        os.chdir(result_dir)
        with open('scenario.json', "r") as f:
            scenario = json.load(f)
        # save the charging buses
        result_dict = {}
        for root, dirs, files, in os.walk(".", topdown=True):
            for file in files:
                # path_lst = file.split(".")
                if 'trans_Th' in file:
                    trans_Th_data = pd.read_csv(file)
        relevant_dcfc_trans = [trans for trans in trans_Th_data.columns if 'dcfc' in trans]
        relevant_L2_trans = [f'trip_trans_{trans.split("_")[-1]}' for trans in scenario['L2_nodes']]
        trans_str_list = relevant_L2_trans + relevant_dcfc_trans
        relevant_Th_data = trans_Th_data[trans_str_list]
        # ref for A and B https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4073181
        A = -27.558
        B = 14573
        # LT = A * np.exp(B/(self.trans_Th + 273.15)) *
        # each transformer is a col in the data
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
