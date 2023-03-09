import os
import copy
import numpy as np
import pandas as pd
import json
from cost_analysis import CostEstimator

import seaborn as sns
import matplotlib.pyplot as plt

# energy_ratings = [8e4, 10e4, 15e4, 20e4, 25e4]
energy_ratings = [5e4, 10e4, 20e4, 40e4, 80e4]
max_c_rates = [0.1, 0.2, 0.5, 1]
# max_c_rates = [0.2, 0.5, 1, 1.5, 2]
# max_c_rates = [0.1, 0.2, 0.5, 1]
plot_font_size = 12


def plot(batt_dtable=None, elec_cost_dtable=None, trans_cost_dtable=None, batt_aging_table=None,
         save_plots_folder=None):
    # sourcery skip: extract-duplicate-method
    sns.heatmap(batt_dtable, cbar_kws={'label': 'cost USD ($)/day'})
    plt.xlabel('Energy (kWh)')
    plt.ylabel('C-rate')
    plt.title("Battery Cost map")
    plt.tight_layout()
    # plt.show()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/battery_cost_map.png')
        plt.close('all')

    sns.heatmap(elec_cost_dtable, cbar_kws={'label': 'cost USD ($)/day'})
    plt.xlabel('Energy (kWh)')
    plt.ylabel('C-rate')
    plt.title("Electricity Cost map")
    plt.tight_layout()
    # plt.show()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/elec_cost_map.png')
        plt.close('all')

    totalcosttable = elec_cost_dtable + batt_dtable
    sns.heatmap(totalcosttable, cbar_kws={'label': 'cost USD ($)/day'})
    plt.xlabel('Energy (kWh)')
    plt.ylabel('C-rate')
    plt.title("Total Cost (electricity + batt) map")
    plt.tight_layout()
    # plt.show()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/elec_batt_cost_map.png')
        plt.close('all')

    sns.heatmap(batt_aging_table, cbar_kws={'label': 'cost USD ($/kWh-day)'})
    plt.xlabel('Energy (kWh)')
    plt.ylabel('C-rate')
    plt.title("Battery aging cost map")
    plt.tight_layout()
    # plt.show()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/battery_aging_map.png')
        plt.close('all')

    sns.heatmap(trans_cost_dtable, cbar_kws={'label': 'Percent Loss of Life (%)/day'})
    plt.xlabel('Energy (kWh)')
    plt.ylabel('C-rate')
    plt.title("Transformer Aging Map")
    plt.tight_layout()
    # plt.show()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/trans_aging_map.png')
        plt.close('all')

    for row in batt_aging_table.T.iterrows():
        row[1].plot.bar()
        plt.title(f'Battery aging for {row[0]}kWh')
        plt.xlabel("C-rate")
        plt.ylabel('Aging cost (USD $/kWh-day)')
        plt.tight_layout()
        if save_plots_folder:
            plt.savefig(f'{save_plots_folder}/batt_aging_{row[0]}kWh.png')
            plt.close()
        # plt.show()


def run_results(case_dir, days_count):
    estimator = CostEstimator(days_count)
    battery_cost = estimator.calculate_battery_cost(case_dir)
    electricity_cost = estimator.calculate_electricity_cost_PGEBEV2s(case_dir, PGE_seperate_file=True)
    # trans_loss_of_life = estimator.calculate_trans_loss_of_life(result_dir)
    print(battery_cost)
    print(electricity_cost)
    # print(trans_loss_of_life)


def collate_results(month):
    main_dir = os.getcwd()
    data_table = pd.DataFrame(
        {energy_rating / 1000: np.zeros(len(max_c_rates)).tolist() for energy_rating in energy_ratings})
    data_table = data_table.set_index(pd.Index(max_c_rates))
    # now go through all files and update table results for both electricity cost and battery costs
    battery_dtable = copy.deepcopy(data_table)
    batt_aging_dtable = copy.deepcopy(data_table)
    electricity_cost_dtable = copy.deepcopy(data_table)
    trans_dtable = copy.deepcopy(data_table)
    for i in range(len(energy_ratings)*len(max_c_rates)):
        resul_dir = f'results/oneshot_{month}{i}'
        os.chdir(resul_dir)
        with open('scenario.json', "r") as f:
            scenario = json.load(f)
            c_rate = scenario['max_c_rate']
            energy = scenario['pack_energy_cap'] / 1000
        with open('postopt_cost_batt.json', "r") as f:
            batt_costs = json.load(f)
            batt_total_cost = batt_costs['battery_sim_0']['total_cost_per_day']
            batt_aging_cost = batt_costs['battery_sim_0']['capital_loss_aging_per_day']
        with open('postopt_cost_charging.json', "r") as f:
            elec_costs = json.load(f)
            elec_total_cost = elec_costs['charging_station_sim_0']['cost_per_day']
            print(elec_costs)
        # with open('postopt_trans_lol.json', "r") as f:
        #     trans_lol = json.load(f)
        #     avg_trans_lol = trans_lol['dcfc_trans_0_LOL_per_day']

        battery_dtable[energy].loc[c_rate] = batt_total_cost
        electricity_cost_dtable[energy].loc[c_rate] = elec_total_cost
        batt_aging_dtable[energy].loc[c_rate] = batt_aging_cost
        # trans_dtable[energy].loc[c_rate] = avg_trans_lol


        os.chdir(main_dir)
    print(electricity_cost_dtable)
    collated_dir = f'{month}_oneshot_collated_results'
    os.mkdir(collated_dir)
    battery_dtable.to_csv(f'{collated_dir}/{month}_battery_costs_per_day.csv')
    electricity_cost_dtable.to_csv(f'{collated_dir}/{month}_elec_costs_per_day.csv')
    (electricity_cost_dtable + battery_dtable).to_csv(f'{collated_dir}/Total_{month}_costs_per_day.csv')
    batt_aging_dtable.to_csv(f'{collated_dir}/{month}_battery_aging_costs_per_day.csv')
    plot(batt_dtable=battery_dtable, elec_cost_dtable=electricity_cost_dtable, batt_aging_table=batt_aging_dtable,
         trans_cost_dtable=trans_dtable, save_plots_folder=collated_dir)


if __name__ == '__main__':
    desired_month = 'June'
    days = 30  # number of days
    for i in range(len(energy_ratings)*len(max_c_rates)):
        # result_dir = f'results/{desired_month}_nomax_{i}'
        result_dir = f'results/oneshot_{desired_month}{i}'
        run_results(result_dir, days)
    collate_results(desired_month)
