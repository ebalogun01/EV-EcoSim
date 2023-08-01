import numpy as np

from cost_analysis import CostEstimator

#%%
num_days = 30
estimator = CostEstimator(num_days)

#%%
start_idx = 0
step = 5
end_idx = 25
for idx in range(start_idx, end_idx, step):
    elec_cost = estimator.calculate_elec_cost(f'results/oneshot_June{idx}')

#%%
import matplotlib.pyplot as plt
elec_prices = np.loadtxt('../elec_rates/PGE_BEV2_S_annual_TOU_rate_15min.csv')[0:96]
load = np.loadtxt('../speechLoadData/speechWeekdayLoad1600.csv')[0, :]
num_steps = 96
fig, ax1 = plt.subplots(figsize=(8, 5))
x_vals = 15 / 60 * np.arange(0, num_steps)
# Plot the first dataset using the first y-axis
ax1.plot(x_vals, load, color='tab:blue')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Power (kW)', color='tab:blue')
ax1.set_ylim(0)
ax1.fill_between(x_vals, load, color='tab:blue', alpha=0.3)
# ax1.tick_params('y', colors='g')

# Create a second y-axis
ax2 = ax1.twinx()

# Plot the second dataset using the second y-axis
ax2.plot(x_vals, elec_prices, '--', color='k', label='TOU price', linewidth=2)
ax2.set_ylim(elec_prices.min())
ax2.set_xlim(0, 23)
ax2.set_ylabel('Price ($/kWh)', color='k')
plt.legend()
# ax2.tick_params('y', colors='b')

# Set title and display the plot
plt.tight_layout()
# plt.title('Plot with Two Y-Axes')
# plt.show()
plt.savefig('TOU_plot.png')

