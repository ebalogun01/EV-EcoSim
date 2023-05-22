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