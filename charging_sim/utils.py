"""Maybe stores general configurations and general functions"""
import cvxpy as cp
# import pandas as pd
# from electricityPrices import PriceLoader
# import numpy as np

day_hours = 24
day_minutes = day_hours * 60
resolution = 15  # minutes
num_homes = 1
num_steps = int(day_minutes / resolution)  # number of time intervals in a day = 96
month_days = {"January": 31, "February": 28, "March": 31, "April": 30, "May": 31, "June": 30, "July": 31,
              "August": 31, "September": 30, "October": 30, "November": 30, "December": 31}

objectives = {'Transformer Aging': [0, 0, 1],
              "Electricity Cost": [1, 0, 0],
              "Battery Degradation": [1, 200, 0],
              'Air Pollution': [],
              "Mixed": [0.1, 1, 0],
              "All": [1, 100, 1]}  # modifies weight placed on multi-obj based on priority/case study

def build_electricity_cost(controller, load, energy_prices_TOU, demand_charge=False):
    # to be used later. For now, keep as-is.
    """Need to update from home load right now; maybe this can be useful in future opt."""
    # TODO: include time-shifting for energy TOU price rates? Add emissions cost pricing based on TOD?
    lam = 10  # this needs to be guided
    # sparsity_cost_factor = 0.000001  # dynamically determine this in future based on load * cost
    # sparsity_cost = cp.norm(controller.battery_power_charge, 1) + \
    #                 cp.norm(controller.battery_power_discharge, 1)
    cost_electricity = cp.sum((cp.multiply(energy_prices_TOU, (load + controller.battery_power -
                                                               controller.solar.battery_power -
                                                               controller.solar.ev_power -
                                                               controller.solar.grid_power))))
    if demand_charge:
        demand_charge_cost = cp.max(cp.pos(load + (controller.battery_power_charge +
                                                   controller.battery_power_discharge -
                                                   controller.solar.battery_power -
                                                   controller.solar.ev_power -
                                                   controller.solar.grid_power)))
        cost_electricity += demand_charge_cost
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


