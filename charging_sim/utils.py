"""Maybe stores general configurations and general functions"""
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

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


def PGE_BEV2_S():
    """Price schedule/TOU rate for PGE EVSE doc can be found here:"""
    peak = 0.39601  # $/kWh
    off_peak = 0.18278  # $/kWh
    super_off_peak = 0.15951  # $/kWh
    peak_times = ["4PM-9PM"]
    off_peak_times = ["12AM-9AM", "2PM-4PM", "9PM-12AM"]
    super_off_peak_times = ["9AM-2PM"]
    hourly_prices = np.zeros((24,))
    hourly_prices = load_prices(peak_times, peak, hourly_prices)
    hourly_prices = load_prices(off_peak_times, off_peak, hourly_prices)
    hourly_prices = load_prices(super_off_peak_times, super_off_peak, hourly_prices)
    times = [f'{int(i)}:00' for i in range(24)]
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})
    plt.xticks(rotation=60, ha="right")
    plt.plot(times, hourly_prices)
    # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.xlabel("Hour of day")
    plt.ylabel("TOU rate ($/kWh)")
    plt.tight_layout()
    plt.show()
    np.save
    return hourly_prices


def load_prices(time_intervals, price, price_vector):
    for interval in time_intervals:
        interval_list = interval.split('-')
        for i in range(len(interval_list)):
            if interval_list[i] == '12AM' and i == 0:
                interval_list[i] = 0
            elif interval_list[i] == '12AM':
                interval_list[i] = 24
            else:
                interval_list[i] = int(interval_list[i][:-2]) if interval_list[i][-2] == 'A' else int(
                    interval_list[i][:-2]) + 12
        price_vector[interval_list[0]:interval_list[1]] = price
    return price_vector


def build_electricity_cost(controller, load, energy_prices_TOU, demand_charge=False):
    # to be used later. For now, keep as-is.
    """Need to update from home load right now; maybe this can be useful in future opt."""
    # TODO: include time-shifting for energy TOU price rates? Add emissions cost pricing based on TOD?
    lam = 10  # this needs to be guided
    # sparsity_cost_factor = 0.000001  # dynamically determine this in future based on load * cost
    sparsity_cost = cp.norm(controller.battery_current_grid, 1) + cp.norm(controller.battery_current_solar, 1) + \
                    cp.norm(controller.battery_current_ev, 1)

    cost_electricity = cp.sum((cp.multiply(energy_prices_TOU, (load + controller.battery_power -
                                                               controller.solar.battery_power -
                                                               controller.solar.ev_power))))
    if demand_charge:
        demand_charge_cost = cp.max(cp.pos(load + (controller.battery_power_charge +
                                                   controller.battery_power_discharge -
                                                   controller.solar.battery_power -
                                                   controller.solar.ev_power)))
        cost_electricity += demand_charge_cost
    return cost_electricity


def build_cost_PGE_BEV2S(controller, load, energy_prices_TOU):
    """This will need to use a heuristic and take the average conservative estimate for gamma"""
    net_load = load + controller.battery_power - controller.solar.battery_power - controller.solar.ev_power
    TOU_cost = cp.sum(cp.multiply(energy_prices_TOU, net_load))
    price_per_block = 95.56  # ($/kW)
    overage_fee = 3.82  # ($/kW)
    charging_block = controller.pge_gamma * 50  # gamma is an integer variable that's at least 1
    subscription_cost = charging_block * price_per_block / month_days["June"]  # This is in blocks of 50kW which makes it very convenient
    penalty_cost = cp.sum(cp.neg(charging_block + net_load) * overage_fee)
    return subscription_cost + penalty_cost + TOU_cost


def build_objective(mode, electricity_cost, battery_degradation_cost, transformer_cost=0):
    """Builds the objective function that we will minimize."""
    lambdas = objectives[mode]
    return cp.sum(
        electricity_cost * lambdas[0]
        + battery_degradation_cost * lambdas[1]
        + transformer_cost * lambdas[2]
    )


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


if __name__ == '__main__':
    PGE_BEV2_S()
