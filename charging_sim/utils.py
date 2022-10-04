"""Maybe stores general configurations and general functions"""
import cvxpy as cp
import pandas as pd
from electricityPrices import PriceLoader
import numpy as np
from solarData import sample_solar

day_hours = 24
day_minutes = day_hours * 60
resolution = 15  # minutes
num_homes = 1
num_steps = int(day_minutes / resolution)  # number of time intervals in a day = 96
month_days = {"January": 31, "February": 28, "March": 31, "April": 30, "May": 31, "June": 30, "July": 31,
              "August": 31, "September": 30, "October": 30, "November": 30, "December": 31}

objectives = {'Transformer Aging': [0, 0, 1],
              "Electricity Cost": [10, 0, 0],
              "Battery Degradation": [1, 200, 0],
              'Air Pollution': [],
              "Mixed": [0.1, 1, 0],
              "All": [1, 100, 1]}  # modifies weight placed on multi-obj based on priority/case study


class EnergyData:
    def __init__(self):
        self._season = "Winter"
        self._home_data_file = "/home/ec2-user/EV50_cosimulation/Datasets/15minute_data_california.csv"

    def get_tou_vector(self):
        if self._season == "Summer":
            return np.repeat(electricityPricesSummer(), 60 / resolution)
        elif self._season == "Winter":
            return np.repeat(electricityPricesWinter(), 60 / resolution)

    # Returns 35040xnum_homes matrix of home energy consumption (kWh)
    @staticmethod
    def get_charging_data():
        load_data = pd.read_csv('/home/ec2-user/EV50_cosimulation/Datasets/USA_CA_San.Francisco.724940_TMY2.csv')['Electricity:Facility [kWh](Hourly)'].\
            to_numpy()
        load_data = np.reshape(np.repeat(load_data, 60 / resolution), (365*num_steps, 1))
        home_data = np.tile(load_data, (1, num_homes))
        return home_data

    def get_pecan_home_data(self):
        load_data = pd.read_csv(self._home_data_file)['grid'].to_numpy()  # not done yet
        load_data = np.reshape(np.repeat(load_data, 60 / resolution), (365 * num_steps, 1))
        home_data = np.tile(load_data, (1, num_homes))
        return home_data

    @staticmethod
    def get_solar_gen():
        return np.reshape(sample_solar(), (365*96, 1))


def build_electricity_cost(controller, load, energy_prices_TOU):
    """Need to update from home load right now; maybe this can be useful in future opt."""
    # TODO: include time-shifting for energy TOU price rates? Add emissions cost pricing based on TOD?
    lam = 10  # this needs to be guided
    sparsity_cost_factor = 0.000 # dynamically determine this in future based on load * cost
    sparsity_cost = cp.norm(controller.battery_power_charge, 1) + cp.norm(controller.battery_power_discharge, 1)
    cost_electricity = cp.sum((cp.multiply(energy_prices_TOU, (load + (controller.battery_power_charge +
                                                                       controller.battery_power_discharge) -
                                                           solar_gen[controller.battery_start:controller.battery_start
                                                            + num_steps])))) + sparsity_cost_factor * sparsity_cost

    # cost_electricity_dem_charge = lam * cp.max((EV_load[battery.start:battery.start + num_steps] +
    #                                             battery.power_charge -
    #                                             battery.power_discharge -
    #                                             solar_gen[battery.start:battery.start + num_steps])) +
    #                                             cost_electricity
    # print(cost_electricity)
    return cost_electricity


def build_objective(mode, electricity_cost, battery_degradation_cost, transformer_cost=0):
    """Builds the objective function that we will minimize."""
    lambdas = objectives[mode]
    obj = cp.sum(electricity_cost * lambdas[0] + battery_degradation_cost * lambdas[1] +
                 transformer_cost * lambdas[2])
    return obj


def add_power_profile_to_object(battery, index, battery_power_profile):
    if 1 <= index <= 31:
        battery.power_profile['Jan'].append(battery_power_profile)
    if 32 <= index <= 59:
        battery.power_profile['Feb'].append(battery_power_profile)
    if 60 <= index <= 90:
        battery.power_profile['Mar'].append(battery_power_profile)
    if 91 <= index <= 120:
        battery.power_profile['Apr'].append(battery_power_profile)
    if 121 <= index <= 151:
        battery.power_profile['May'].append(battery_power_profile)
    if 152 <= index <= 181:
        battery.power_profile['Jun'].append(battery_power_profile)
    if 182 <= index <= 212:
        battery.power_profile['Jul'].append(battery_power_profile)
    if 213 <= index <= 233:
        battery.power_profile['Aug'].append(battery_power_profile)
    if 234 <= index <= 263:
        battery.power_profile['Sep'].append(battery_power_profile)
    if 264 <= index <= 294:
        battery.power_profile['Oct'].append(battery_power_profile)
    if 295 <= index <= 334:
        battery.power_profile['Nov'].append(battery_power_profile)
    if 335 <= index <= 365:
        battery.power_profile['Dec'].append(battery_power_profile)


def plot_control_actions(controller):
    actions = controller.actions


def show_results(savings_total, battery_object, energy_price, EV_load):
    """This is for calculating cost savings etc"""
    num_cells = battery_object.topology[2]
    battery_power = np.reshape(battery_object.control_current, (num_steps, 1))
    solar = solar_gen[battery_object.start:battery_object.start + num_steps]

    travel_dollars = 0
    initial_cost = travel_dollars + resolution / 60 * np.sum(energy_price * EV_load)
    print("Initial energy cost is ${}".format(initial_cost))
    net_cost_opt_onlySolar = travel_dollars + resolution / 60 * np.sum(energy_price * (EV_load -
                    solar_gen[battery_object.start:battery_object.start + num_steps]))
    print("Energy cost with just solar is ${}".format(net_cost_opt_onlySolar))
    net_cost_opt =  resolution / 60 * np.sum(energy_price * (EV_load + num_cells*battery_power - solar))
    savings = initial_cost - net_cost_opt
    savings_total += savings
    print("Energy cost after arbitrage is ${}".format(net_cost_opt))
    print("Today's savings is ${}".format(initial_cost - net_cost_opt))
    print("Running savings is: $", savings_total)
    battery_object.savings = savings_total
    return savings_total


power_data = EnergyData()  # Residential energy consumption
solar_rating = num_homes  # 1 kW rating per num_charging_stations
solar_gen = solar_rating * power_data.get_solar_gen()  # Solar generation at 15 min intervals (35040x1 array)
