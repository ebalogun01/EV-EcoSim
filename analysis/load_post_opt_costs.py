"""
This module calculates the levelized cost of energy and populates into tables/cost matrices, which are saved in the
respective files and folders.
"""

import os
import copy
import numpy as np
import pandas as pd
import json
from cost_analysis import CostEstimator
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from charging_sim.utils import MONTHS_LIST


# Load the user_input dict state.

USR_INPUT_PATH = '../dash-fe/input/sim_run_settings.json'
with open(USR_INPUT_PATH, "r") as f:
    USR_INPUT_DICT = json.load(f)

ENERGY_RATINGS = USR_INPUT_DICT['battery']['pack_energy_cap']
# ENERGY_RATINGS = [ENERGY_RATINGS_WH[i]/1000 for i in range(len(ENERGY_RATINGS_WH))]
MAX_C_RATES = USR_INPUT_DICT['battery']['max_c_rate']
print(ENERGY_RATINGS, MAX_C_RATES)
PLOT_FONT_SIZE = 16
matplotlib.rc('font', **{'size': PLOT_FONT_SIZE})


def plot_tables(batt_dtable=None, elec_cost_dtable=None, trans_cost_dtable=None, batt_aging_table=None,
                solar_cost_table=None, save_plots_folder: str = None):
    """
    Plots the tables into a visualized results matrix and plots a stacked bar chart.

    :param batt_dtable: Data table for battery costs.
    :param elec_cost_dtable: Dataframe for electricity costs.
    :param trans_cost_dtable: Dataframe for transformer costs.
    :param batt_aging_table: Dataframe for battery aging costs.
    :param solar_cost_table: Dataframe for solar cost (LCOE).
    :param save_plots_folder: Directory in which plots are saved.
    :return: None.

    """
    sns.heatmap(batt_dtable, cbar_kws={'label': 'cost USD ($)/kWh'})
    plt.xlabel('Battery Energy Capacity (kWh)')
    plt.ylabel('C-rate')
    plt.title("Battery Cost map")
    plt.tight_layout()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/battery_cost_map.png')
        plt.close('all')

    sns.heatmap(elec_cost_dtable, cbar_kws={'label': 'cost USD ($)/kWh'})
    plt.xlabel('Battery Energy Capacity (kWh)')
    plt.ylabel('C-rate')
    plt.title("Electricity Cost map")
    plt.tight_layout()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/elec_cost_map.png')
        plt.close('all')

    totalcosttable = elec_cost_dtable + batt_dtable
    sns.heatmap(totalcosttable, cbar_kws={'label': 'cost USD ($)/kWh'})
    plt.xlabel('Battery Energy Capacity (kWh)')
    plt.ylabel('C-rate')
    plt.title("Total Cost (electricity + batt) map")
    plt.tight_layout()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/elec_batt_cost_map.png')
        plt.close('all')

    sns.heatmap(batt_aging_table, cbar_kws={'label': 'cost USD ($/kWh)'})
    plt.xlabel('Battery Energy Capacity (kWh)')
    plt.ylabel('C-rate')
    plt.title("Battery aging cost map")
    plt.tight_layout()
    if save_plots_folder:
        plt.savefig(f'{save_plots_folder}/battery_aging_map.png')
        plt.close('all')

    if trans_cost_dtable:
        sns.heatmap(trans_cost_dtable, cbar_kws={'label': 'Percent Loss of Life (%)/day'})
        plt.xlabel('Battery Energy Capacity (kWh)')
        plt.ylabel('C-rate')
        plt.title("Transformer Aging Map")
        plt.tight_layout()
        if save_plots_folder:
            plt.savefig(f'{save_plots_folder}/trans_aging_map.png')
            plt.close('all')

    for row in batt_aging_table.T.iterrows():
        row[1].plot.bar()
        plt.title(f'Battery aging for {row[0]}kWh')
        plt.xlabel("C-rate")
        plt.ylabel('Aging cost (USD $/kWh)')
        plt.tight_layout()
        if save_plots_folder:
            plt.savefig(f'{save_plots_folder}/batt_aging_{row[0]}kWh.png')
            plt.close()

    plot_stacked_bar(elec_cost_dtable, batt_dtable, solar_costs=solar_cost_table, save_plot_path=save_plots_folder)


def plot_stacked_bar(elec_costs, batt_costs, solar_costs=None, save_plot_path=""):
    """
    Plots a stacked bar to visualize the portion of overall LCOE each cost component contributes to the system.

    :param elec_costs: Dataframe of electricity costs.
    :param batt_costs: Dataframe of battery levelized costs.
    :param solar_costs: Dataframe of levelized solar costs.
    :param str save_plot_path: String to path for which results are saved.
    :return: None.
    """
    plot_font_size = 16
    font = {'size': plot_font_size}
    matplotlib.rc('font', **font)
    capacities = [f'{str(size)}' for size in elec_costs.columns]
    c_rate_idx = 0  # ONLY CHANGE THIS IF LOOKING AT OTHER C-RATES
    # todo: set this to work for multiple c-rates
    cost_component = {}
    if solar_costs is not None:
        cost_component['Solar'] = solar_costs.to_numpy()[c_rate_idx]
    cost_component['Battery'] = batt_costs.to_numpy()[c_rate_idx]
    cost_component['Electricity'] = elec_costs.to_numpy()[c_rate_idx]

    width = 0.6  # the width of the bars: can also be len(x) sequence
    neg_cost = False  # need to build a more robust solution to this in the future
    if np.any(cost_component['Electricity'] < 0):
        neg_cost = True  # this is mainly for plotting aesthetics
    fig, ax = plt.subplots()
    bottom = np.zeros(len(capacities))

    for name, cost in cost_component.items():
        if neg_cost and name == 'Electricity':
            bottom = np.zeros(len(capacities))
        p = ax.bar(capacities, cost, label=name, bottom=bottom)
        bottom += cost
    plt.ylabel("LCOE ($/kWh)")
    plt.xlabel("Battery Energy Capacity (kWh)")
    ax.set_title('LCOE breakdown')
    ax.legend(loc='upper right')
    fig.tight_layout()

    if save_plot_path:
        plt.savefig(f'{save_plot_path}/total_expediture.png')
        plt.close('all')

    fig, ax = plt.subplots()
    net_cost = np.zeros(len(capacities))
    for name, cost in cost_component.items():
        net_cost += cost
    p = ax.bar(capacities, net_cost)
    ax.set_title('LCOE (system + elec)')
    plt.ylabel("LCOE ($/kWh)")
    plt.xlabel("Battery Energy Capacity (kWh)")
    fig.tight_layout()
    if save_plot_path:
        plt.savefig(f'{save_plot_path}/net_expediture.png')
        plt.close('all')


def run_results(case_dir, days_count, batt_cost: bool = True, elec_cost: bool = True, trans_cost: bool = False,
                oneshot=False):
    """
    Uses the CostEstimator class to calculate the cost for each case and saves it into the case directory file.

    :param str case_dir: Directory for a given case/scenario.
    :param int days_count: Number of days to calculate (usually 30 for now).
    :param bool batt_cost: Boolean to decide if battery cost is calculated.
    :param bool elec_cost: Boolean to decide if electricity cost is calculated.
    :param bool trans_cost: Boolean to decide if transformer cost is calculated.
    :return: None.
    """
    estimator = CostEstimator(days_count)
    #   Calculated values are populated in their respective scenario directories.
    if batt_cost:
        estimator.calculate_battery_cost(case_dir)
    if elec_cost:
        estimator.calculate_electricity_cost_PGEBEV2s(case_dir, PGE_separate_file=oneshot)
        # not to add option into this for easy toggling
    if trans_cost:
        estimator.calculate_trans_loss_of_life(case_dir)


def collate_results(month, solar=True, trans=True, oneshot=False):
    """
    Collates the results for each scenario and saves them in a result matrix.

    :param month: Month for which the results are being collated
    :param bool solar: Boolean to decide if to include LCOE of solar.
    :param bool trans: Boolean to decide if transformer aging matrix will be included.
    :param bool oneshot: Boolean to tell the function if the results were obtained from oneshot or mpc simulation.
    :return: None
    """
    solar_lcoe = 0
    if solar:
        solar_lcoe = 0.067
    main_dir = os.getcwd()
    data_table = pd.DataFrame(
        {energy_rating / 1000: np.zeros(len(MAX_C_RATES)).tolist() for energy_rating in ENERGY_RATINGS})
    data_table = data_table.set_index(pd.Index(MAX_C_RATES))
    # Now go through all files and update table results for both electricity cost and battery costs.
    battery_dtable = copy.deepcopy(data_table)
    batt_aging_dtable = copy.deepcopy(data_table)
    electricity_cost_dtable = copy.deepcopy(data_table)
    trans_dtable = copy.deepcopy(data_table)
    solar_dtable = copy.deepcopy(data_table)
    for i in range(len(ENERGY_RATINGS) * len(MAX_C_RATES)):
        if oneshot:
            resul_dir = f'results/oneshot_{month}{i}'
        else:
            resul_dir = f'results/{month}{i}'
        os.chdir(resul_dir)
        with open('scenario.json', "r") as f:
            scenario = json.load(f)
            c_rate = scenario['battery']['max_c_rate']
            energy = scenario['battery']['pack_energy_cap'] / 1000
        with open('postopt_cost_batt.json', "r") as f:
            batt_costs = json.load(f)
            batt_total_cost = batt_costs['battery_sim_0']['lcoe']
            batt_aging_cost = batt_costs['battery_sim_0']['lcoe_aging']
        with open('postopt_cost_charging.json', "r") as f:
            elec_costs = json.load(f)
            elec_total_cost = elec_costs['charging_station_sim_0']['cost_per_kWh']
        avg_trans_lol = 0
        if trans:
            with open('postopt_trans_lol.json', "r") as f:
                trans_lol = json.load(f)
                avg_trans_lol = trans_lol['dcfc_trans_0_LOL_per_day']

        battery_dtable[energy].loc[c_rate] = batt_total_cost
        electricity_cost_dtable[energy].loc[c_rate] = elec_total_cost
        batt_aging_dtable[energy].loc[c_rate] = batt_aging_cost

        solar_dtable[energy].loc[c_rate] = solar_lcoe  # dollars/kWh
        trans_dtable[energy].loc[c_rate] = avg_trans_lol
        os.chdir(main_dir)
    print("Electricity levelized cost table\n", electricity_cost_dtable)
    if oneshot:
        collated_dir = f'{month}_oneshot_collated_results'
    else:
        collated_dir = f'{month}_mpc_collated_results'
    if not os.path.isdir(collated_dir):
        os.mkdir(collated_dir)
    battery_dtable.to_csv(f'{collated_dir}/{month}_battery_costs_per_day.csv')
    electricity_cost_dtable.to_csv(f'{collated_dir}/{month}_elec_costs_per_day.csv')
    (electricity_cost_dtable + battery_dtable + solar_dtable).to_csv(f'{collated_dir}/Total_{month}_costs_per_day.csv')
    batt_aging_dtable.to_csv(f'{collated_dir}/{month}_battery_aging_costs_per_day.csv')
    if trans:
        trans_dtable.to_csv(f'{collated_dir}/{month}_trans_aging_per_day.csv')
    else:
        trans_dtable = None
    plot_tables(batt_dtable=battery_dtable, elec_cost_dtable=electricity_cost_dtable,
                batt_aging_table=batt_aging_dtable,
                trans_cost_dtable=trans_dtable, solar_cost_table=solar_dtable, save_plots_folder=collated_dir)


def run():
    print(USR_INPUT_DICT['month']-1)
    desired_month = MONTHS_LIST[USR_INPUT_DICT['month']-1]
    print('Running analysis for month: ', desired_month)
    oneshot = True
    include_trans = False
    days = USR_INPUT_DICT['num_days']  # number of days
    # Change directory to working directory for this file
    os.chdir('../analysis')
    for i in range(0, len(ENERGY_RATINGS) * len(MAX_C_RATES)):
        if oneshot:
            result_dir = f'results/oneshot_{desired_month}{i}'
        else:
            result_dir = f'results/{desired_month}{i}'
        run_results(result_dir, days, trans_cost=include_trans, oneshot=oneshot)
    collate_results(desired_month, trans=include_trans, oneshot=oneshot)


# RUN THIS FILE
# if __name__ == '__main__':
#     run()
