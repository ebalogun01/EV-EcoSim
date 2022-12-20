import multiprocessing as mp
import sys
import gblvar
# import time

if not gblvar.charging_sim_path_append:
    sys.path.append('../../../EV50_cosimulation/charging_sim')  # change this
    gblvar.charging_sim_path_append = True
    print('append 1')


# RUN TYPE
sequential_run = False
parallel_run = False
single_run = True

# BATTERY SCENARIOS
num_vars = 6
min_power = 0
max_power = 0
power_ratings = []  # this should be redundant for max_c_rate
energy_ratings = [80, 100, 150, 200, 250]
max_c_rates = [0.2, 0.5, 1, 1.5, 2]
min_SOCs = [0.1, 0.2, 0.3]
max_SOCs = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


def make_scenarios():
    scenarios_list = []
    idx = 0
    for Er in energy_ratings:
        for c_rate in max_c_rates:
            scenario = {'pack_energy_cap': Er, 'max_c_rate': c_rate, 'index': idx}
            scenarios_list.append(scenario)
            idx += 1
    return scenarios_list


def run(scenario):
    import master_sim
    master_sim.run(scenario)


def run_scenarios_parallel():
    scenarios = make_scenarios()
    num_cores = mp.cpu_count()
    if num_cores > 1:
        use_cores_count = num_cores - 2  # leave one out
        print("Running {} parallel scenarios...".format(use_cores_count))
        pool = mp.Pool(use_cores_count)
        pool.map(run, [scenarios[i] for i in range(num_cores)])


def run_scenarios_sequential():
    scenarios = make_scenarios()
    for scenario in scenarios:
        import master_sim
        master_sim.run(scenario)
        # time.sleep(5)


def run_scenario_single():
    """This function just runs one scenario"""
    # Keep changing this for each run
    Er_idx = 2
    c_rate_idx = 1
    idx = 11
    scenario = {'pack_energy_cap': energy_ratings[Er_idx],
                'max_c_rate': max_c_rates[c_rate_idx],
                'index': idx}
    run(scenario)


if __name__ == '__main__':
    if sequential_run:
        run_scenarios_sequential()
    elif single_run:
        run_scenario_single()
    else:
        run_scenarios_parallel()
