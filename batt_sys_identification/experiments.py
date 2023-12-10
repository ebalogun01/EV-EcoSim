from battery_identification import BatteryParams
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_trials_paper(path, NUM_TRIALS=10):
    """
    Runs the system identification algorithm for a number of trials and saves the results to file. This runs the trials
    used in the battery system identification portion of the paper.

    :return: None.
    """
    for i in range(NUM_TRIALS):
        print(f'Running trial {i + 1} of {NUM_TRIALS}...')
        battery_data = pd.read_csv(path)
        m = BatteryParams(battery_data)
        m.run_sys_identification(use_initial_pop=False, cell_name=path.split('_')[4], diagn=i + 1, save=True)
        m.plot_correction_scheme_comparison()
        print(f'Done with trial {i + 1} of {NUM_TRIALS}...')
        print('-------------------------------------------')


def make_error_vector(cell_name, num_trials):
    """
    Creates the error vectors for the plots included in the paper. (See battery system identification section of the
    paper).

    :param cell_name: Name of the cell for which data is provided (See naming convention).
    :param num_trials: Number of trials that were run to during the fit testing
    .
    :return Tuple(List): Tuple of lists containing the error vectors or mean absolute percent error (MAPE), mean squared
    error (MSE), and maximum absolute percent error (max APE).

    """
    mape_list = [np.loadtxt(f'0_MAPE_{cell_name}_{i + 1}.csv') * 100 for i in range(num_trials)]
    mse_list = [np.loadtxt(f'0_MSE_{cell_name}_{i + 1}.csv') * 100 for i in range(num_trials)]
    max_ape_list = [np.loadtxt(f'0_max_APE_{cell_name}_{i + 1}.csv') * 100 for i in range(num_trials)]

    return mape_list, mse_list, max_ape_list


def create_ro_plot():
    """
    Creates Ro vs. SoC plot for the paper.

    :return: None
    """
    data_path = 'batt_iden_test_data_W8_1.csv'
    batt_data = pd.read_csv(data_path)
    module = BatteryParams(batt_data)
    params = np.loadtxt('battery_params_W9_4.csv', delimiter=',')
    module.params = params
    module.plot_Ro()


def create_error_boxplots(use_grid=False):
    """
    Creates the box plots used in the original paper. This is for the error statistics and non-functional
    to the main functionality of the package.

    :return: None.
    """
    # Assuming you have error data for W8, W9, and W10
    NUM_TRIALS = 10
    error_data_w8, _, _ = make_error_vector('W8', NUM_TRIALS)  # returns mse, MAPE, max_ape respectively
    error_data_w9, _, _ = make_error_vector('W9', NUM_TRIALS)  # returns mse, MAPE, max_ape respectively
    error_data_w10, _, _ = make_error_vector('W10', NUM_TRIALS)  # returns mse, MAPE, max_ape respectively

    # Combine the data into a list
    error_data = [error_data_w8, error_data_w9, error_data_w10]

    # Set up a color palette for the boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create a box plot with custom styling
    fig, ax = plt.subplots()
    box = ax.boxplot(error_data, patch_artist=True, labels=['W8', 'W9', 'W10'], showmeans=True, meanline=True)

    # Customize box colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    for mean_line in box['means']:
        mean_line.set(color='black', linewidth=2)

    for median_line in box['medians']:
        median_line.set(color='yellow', linewidth=1.2)

    if use_grid:
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Set labels and title
    ax.set_title('Battery System Identification errors', fontsize=14)
    ax.set_xlabel('Battery Samples', fontsize=14)
    ax.set_ylabel('Mean Absolute Percent Error (%)', fontsize=14)

    # Customize font size for tick labels
    ax.tick_params(axis='both', labelsize=14)

    # Add legend
    legend_labels = ['W8', 'W9', 'W10']
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=14) for color in
                      colors]
    ax.legend(legend_patches, legend_labels, loc='upper center')
    fig.tight_layout()

    # Show the plot
    plt.savefig('battery_model_error_mape.png', dpi=300)
    plt.show()


def run():
    # These are for experiment purposes only. They are not used in the main functionality of the package.
    import multiprocessing as mp
    # List all the csv files in current directory and let user choose one.
    # User uploaded data will be saved in the current directory as a temp_data.csv file.
    data_paths = [
        'batt_iden_test_data_W8_1.csv',
        'batt_iden_test_data_W9_1.csv',
        'batt_iden_test_data_W10_1.csv'
    ]
    use_cores_count = min(mp.cpu_count(), len(data_paths))

    with mp.get_context("spawn").Pool(use_cores_count) as pool:
        print(f"Running {use_cores_count} parallel battery trial processes")
        pool.map(run_trials_paper, data_paths)

    # Create the plots for the paper
    create_error_boxplots()
    create_ro_plot()
