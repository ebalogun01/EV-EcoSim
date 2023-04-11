from cost_analysis import CostEstimator

#%%
num_days = 30
estimator = CostEstimator(num_days)

#%%
elec_cost = estimator.calculate_elec_cost('results/oneshot_June0')