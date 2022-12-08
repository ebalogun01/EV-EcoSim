import matplotlib.pyplot as plt
# import cvxpy as cp


def plot_results(test_demand, battery, solar):
    plt.figure()
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.set_xlabel('15 min Intervals')
    ax1.set_ylabel('Demand [kW]')
    ax2.set_ylabel('State of Charge')
    ax1.plot(test_demand)
    after_storage_demand = test_demand + battery.topology[2]*battery.true_power - solar
    energy_only_solar = test_demand - solar
    ax1.plot(after_storage_demand, marker='o', linewidth=2, linestyle='dashed')
    ax1.plot(energy_only_solar)
    # ax1.plot(power_travel)
    ax1.legend(['Demand Curve before battery storage', 'Demand curve with battery and Solar',
                'Solar without Battery'], loc="upper left")
    # ax2.plot(battery.true_SOC, color='k')
    ax2.plot(battery.SOC.value , color='r')
    ax2.legend(['State of Charge'], loc="lower left")

    plt.grid(axis='y', alpha=0.75)
    plt.title('Optimization Performance test')

    plt.show()

