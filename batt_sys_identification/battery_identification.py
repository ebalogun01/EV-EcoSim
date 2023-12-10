"""
This module hosts the Battery System Identification class used for fitting battery ECM model parameters from data.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

import pandas as pd
import pygad
import cvxpy as cp


class BatteryParams:
    """
    Battery system identification class with open circuit voltage correction scheme.
    This class takes in dataframe or csv with some given fields during instantiation.\n

    Dataframe fields (columns) must include the following literally and case-sensitive:

    * current - battery/cell current time-series.
    * voltage - corresponding battery/cell voltage time-series.
    * soc - corresponding battery/cell state of charge time-series.
    * ocv - corresponding battery/cell open circuit voltage time-series.

    How to use:

    * data = pd.read_csv(data_path). This loads a pandas dataframe.
    * module = BatteryParams(data)
    * module.run_sys_identification()
    * module.plot_correction_scheme_comparison()

    Writes new corrected open-circuit voltages and battery parameters to file within the folder. Can be downloaded via
    the web-tool.

    :param data: Battery test data from which to fit the identification params.
    """
    def __init__(self, data):
        self.data = data
        self.current = data.current.values
        self.voltage = data.voltage.values
        self.ocv = data.ocv.values
        self.ocv_no_corr = data.ocv.values     # Uncorrected open circuit voltage.
        self.soc = data.soc.values
        self.params = []
        self.params_uncorr = []
        self.init_params = []   # Can be initialized later.
        self._default_population = [
            [1.058596e-02, 5.93096e-02, -4.17993e+00, 1.58777e-02, 9.75325e+03, 7.45710e-03, 4.01823e+03],
            [3.43209856e-02, 1.74575496e-02, -4.29315177e+00, 2.01081488e-04, 9.56317565e+03, 2.12122173e-03, 9.57647155e+03],
            [2.30552317e-02, 8.44786075e-02, -3.99474379e+00, 2.99153462e-03, 8.87442972e+03, 2.30526762e-03, 9.05492854e+03],
            [-5.82959510e-03, 5.20569661e-02, -2.18763704e-01, 8.67466142e-04, 7.05178934e+03, 9.52035315e-03, 6.81286695e+03],
            [5.72212881e-02, -1.43378662e-02, -6.12054235e-01, 4.80304568e-03, 5.45105883e+03, 4.74474171e-04, 7.31264565e+03],
            [2.69767183e-03, 9.62504736e-02, -1.29365965e+00, 6.11142914e-03, 9.80086309e+03, 1.69531170e-03, 8.00324801e+03],
            [1.16897734e-02, 4.54745088e-02, -3.28524770e+00, 1.35089385e-02, 5.69817778e+03, 3.98807635e-03, 8.31128396e+03],
            [1.17640691e-02, 7.63076005e-02, -3.66411278e+00, 1.23863884e-02, 9.30697417e+03, 4.56975167e-04, 5.57718908e+03],
            [1.93628327e-03, 5.21192971e-02, -1.96503048e+00, 1.77612540e-02, 5.47448933e+03, 6.42417508e-03, 8.05424318e+03],
            [6.78756010e-03, 2.96217428e-02, -2.03289183e+00, 1.88927411e-02, 1.91748586e+03, 9.10005523e-03, 5.14585687e+03]
        ]

    def _assert_data_len(self):
        """
        Checks the imported data to ensure all fields have the same length.

        :return: None.
        """
        assert self.voltage.shape[0] == self.soc.shape[0]
        assert self.soc.shape[0] == self.current.shape[0]
        assert self.soc.shape[0] == self.ocv.shape[0]

    def _cost_fun_pygad(self, ga_instance, x, idx):
        """
        Provides the cost function used by PyGAD for parameter search.

        :param object ga_instance: Genetic Algorithm instance (pygad).
        :param x: ndarray/vector of parameters $\theta$.
        :param idx: default param used by ga_instance to run the problem
        :return: Cost function, as described in EV-Ecosim paper.
        """
        cost = 0
        V_data = self.voltage
        data_len = V_data.shape[0]
        A_Ro = x[0]
        B_Ro = x[1]
        C_Ro = x[2]
        R1 = x[3]
        C1 = x[4]
        R2 = x[5]
        C2 = x[6]
        dt = 1

        Ro = B_Ro * np.exp(C_Ro * self.soc) + A_Ro * np.exp(self.soc)  # updated the new Ro based on intuition
        I_t = -self.current  # this will be loaded from .npy

        I_R1 = np.zeros((I_t.shape[0],))
        I_R2 = np.zeros((I_t.shape[0],))
        for j in range(1, len(I_R1)):
            I_R1[j] = np.round(np.exp(-dt / (R1 * C1)) * I_R1[j - 1] + (1 - np.exp(-dt / (R1 * C1))) * I_t[j - 1],
                               10)
            I_R2[j] = np.round(np.exp(-dt / (R2 * C2)) * I_R2[j - 1] + (1 - np.exp(-dt / (R2 * C2))) * I_t[j - 1],
                               10)
        V_t = self.ocv - np.multiply(I_t, Ro) - np.multiply(I_R1, R1) - np.multiply(I_R2, R2)
        assert V_t.shape == self.voltage.shape
        cost += np.sum(np.square(V_t - V_data))
        cost = np.sqrt(cost / data_len)
        if Ro.min() <= 0:
            cost += 100
        if Ro[0] > Ro[-1]:
            cost += 1
        return -10 * cost

    def _init_population(self):
        """
        Initializes the PyGAD population. This is used to use a preset population as a warm-start. This helps the
        parameter search to converge much quickly than starting from random genes within desired bounds.

        :return:
        """
        print('Initializing population for warm-start...')
        # f = open('init_pop.txt', 'r')
        # initial_params = f.read()
        # f.close()
        # self.init_params = ast.literal_eval(initial_params)
        self.init_params = self._default_population
        print('Done initializing population.')

    @staticmethod
    def _get_pygad_bounds():
        """
        Returns gene space range, representing the minimum and maximum values for each gene in the parameter vector.

        :return list gene_space_range: List of dicts providing the range for each gene.
        """
        # todo: Include option to change the bounds for the parameters.
        lb2 = [-0.1, -0.1, -5, 0.0001, 100, 0.0004, 4000]
        ub2 = [0.1, 0.1, 0, 0.02, 10000, 0.01, 10000]

        lb3 = [-10, -10, -10, 0.00004, 0.03, 0.0004, 600]  # change the range for testing capacitance variation with SOC
        # if any
        ub3 = [10, 10, 10, 0.01, 0.06, 0.01, 10000]

        lb = [0.02, 0.0001, 100, 0.00004, 900]
        ub = [0.04, 0.01, 10000, 0.004, 10000]
        gene_space_range = [{'low': low, 'high': high} for low, high in zip(lb2, ub2)]
        return gene_space_range

    def ga(self, num_generations=40, num_parents_mating=2, sol_per_pop=10, num_genes=7, crossover_type="single_point",
           mutation_type="adaptive", parent_selection_type="sss", mutation_percent_genes=60,
           mutation_prob=(0.3, 0.1), crossover_prob=None):
        """
        Runs the genetic algorithm instance. Please see PyGAD official documentation for more explanation of
        fields/params.
        The default parameters have been selected to optimize accuracy and speed, however, any user may find a
        combination of params that work better for a given set of battery data.

        :param crossover_prob:
        :param num_generations: Number of generations. Default is 100.
        :param num_parents_mating: Number of parents to combine to form the next offspring. Default is 2.
        :param sol_per_pop: Number of solutions per population (a set of genes is one solution), or offspring size.
        :param num_genes: Equivalent to the number of parameters being searched
        :param crossover_type: Describes how the cross-over between mating genes is done.
        :param mutation_type: Describes gene mutation. Default adaptive
        :param parent_selection_type: Parent selection scheme for the next population.
        :param mutation_percent_genes: The percentage of genes that undergo mutation.
        :param mutation_prob: The probability of selecting a gene for applying the mutation operation.
        Its value must be between 0.0 and 1.0 inclusive.
        :return: Solution vector of optimized parameters.
        """
        gene_space_range = self._get_pygad_bounds()     # Get the gene space range.
        if self.init_params:
            ga_instance = pygad.GA(num_generations=num_generations,
                                   initial_population=self.init_params,
                                   num_parents_mating=num_parents_mating,
                                   fitness_func=self._cost_fun_pygad,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   mutation_type=mutation_type,
                                   mutation_probability=mutation_prob,
                                   mutation_percent_genes=mutation_percent_genes,
                                   parent_selection_type=parent_selection_type,
                                   crossover_type=crossover_type,
                                   allow_duplicate_genes=False,
                                   crossover_probability=crossover_prob,
                                   stop_criteria=["reach_0.05", "saturate_20"],
                                   gene_space=gene_space_range,
                                   parallel_processing=4)
        else:
            ga_instance = pygad.GA(num_generations=num_generations,
                                   num_parents_mating=num_parents_mating,
                                   fitness_func=self._cost_fun_pygad,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   mutation_type=mutation_type,
                                   mutation_probability=mutation_prob,
                                   mutation_percent_genes=mutation_percent_genes,
                                   parent_selection_type=parent_selection_type,
                                   crossover_type=crossover_type,
                                   allow_duplicate_genes=False,
                                   crossover_probability=crossover_prob,
                                   stop_criteria=["reach_0.05", "saturate_20"],
                                   gene_space=gene_space_range,
                                   parallel_processing=4)

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        ga_instance.plot_fitness()
        print(f'Parameters of the best solution : {solution}')
        print(f"Fitness value of the best solution = {solution_fitness}")
        return solution

    def run_sys_identification(self, cell_name='0', diagn: int = 0, use_initial_pop: bool = True,
                               quadratic_bias: bool = True, save: bool = True, error_stats: bool=True) -> None:
        """
        Runs the GA for system identification.

        :return:
        """
        if use_initial_pop:
            self._init_population()
        self.params = self.ga()

        if quadratic_bias:
            self.run_ocv_correction(cell_name=cell_name, use_quadratic=True)
        else:
            self.run_ocv_correction(cell_name=cell_name)   # Uses simple linear bias correction.
        # Run again.
        self.params = self.ga()     # New params.
        np.savetxt(f'battery_params_{cell_name}_{diagn}.csv', self.params, delimiter=',')
        if error_stats:
            self._calculate_error(save=save, cell_name=cell_name, diagn=diagn)

    def run_ocv_correction(self, use_quadratic=False, cell_name='0', diagn=0):
        """
        Fits the parameters for the open circuit voltage correction scheme. Updates the ocv attribute. The open circuit
        voltage correction scheme can be quadratic or linear. The quadratic scheme was used in the original paper.

        :return: None.
        """
        self._validate_params()
        self.params_uncorr = copy.copy(self.params)
        x = self.params
        V_data = self.voltage
        A_Ro = x[0]
        B_Ro = x[1]
        C_Ro = x[2]
        R1 = x[3]
        C1 = x[4]
        R2 = x[5]
        C2 = x[6]
        dt = 1

        Ro = B_Ro * np.exp(C_Ro * self.soc) + A_Ro * np.exp(self.soc)  # updated the new Ro based on intuition
        I_t = -self.current  # this will be loaded from .npy
        I_R1 = np.zeros((I_t.shape[0],))
        I_R2 = np.zeros((I_t.shape[0],))

        for j in range(1, len(I_R1)):
            I_R1[j] = np.round(np.exp(-dt / (R1 * C1)) * I_R1[j - 1] + (1 - np.exp(-dt / (R1 * C1))) * I_t[j - 1], 10)
            I_R2[j] = np.round(np.exp(-dt / (R2 * C2)) * I_R2[j - 1] + (1 - np.exp(-dt / (R2 * C2))) * I_t[j - 1], 10)
        V_t = self.ocv - np.multiply(I_t, Ro) - np.multiply(I_R1, R1) - np.multiply(I_R2, R2)
        beta = cp.Variable(1)
        const_bias = cp.Variable(1)

        if use_quadratic:
            ocv_corr = self.ocv**2 * beta + const_bias     # work on improving the representation of this ocv correction term
        else:
            ocv_corr = self.ocv * beta + const_bias
        V_t2 = ocv_corr - np.multiply(I_t, Ro) - np.multiply(I_R1, R1) - np.multiply(I_R2, R2)
        obj = cp.Minimize(cp.sum(cp.abs(V_t2 - V_data)))
        problem = cp.Problem(obj)
        problem.solve(solver=cp.ECOS, verbose=False)
        print("OCV correction beta: ",  beta.value)
        print("OCV correction const. bias: ", const_bias.value)
        ocv_bias_correction_vector = np.array([beta.value, const_bias.value])
        np.savetxt('../batt_sys_identification/OCV_bias_correction_params_{}_{}.csv'.format(cell_name, diagn),
                   ocv_bias_correction_vector)
        self.ocv = ocv_corr.value
        self.data['ocv_corr'] = self.ocv
        self.data.to_csv('../batt_sys_identification/input_data_with_ocv_corr_voltage_{}_{}.csv'.
                         format(cell_name, diagn))
        #    Adds corrected ocv as field and writes the input data

    def _calculate_error(self, save=True, cell_name: str = '0', diagn: int = 0, trial: int = 0):
        """
        Calculates the error between the model and the training data.

        :param int trial: Trial number.
        :param int diagn: Diagnosis number.
        :param str cell_name: Cell name.
        :param bool save: Whether to save the error statistics to file.

        :return: Error.
        """
        x = self.params
        V_data = self.voltage
        A_Ro = x[0]
        B_Ro = x[1]
        C_Ro = x[2]
        R1 = x[3]
        C1 = x[4]
        R2 = x[5]
        C2 = x[6]
        dt = 1

        Ro = B_Ro * np.exp(C_Ro * self.soc) + A_Ro * np.exp(self.soc)
        I_t = -self.current
        I_R1 = np.zeros((I_t.shape[0],))
        I_R2 = np.zeros((I_t.shape[0],))

        for j in range(1, len(I_R1)):
            I_R1[j] = np.round(np.exp(-dt / (R1 * C1)) * I_R1[j - 1] + (1 - np.exp(-dt / (R1 * C1))) * I_t[j - 1], 10)
            I_R2[j] = np.round(np.exp(-dt / (R2 * C2)) * I_R2[j - 1] + (1 - np.exp(-dt / (R2 * C2))) * I_t[j - 1], 10)
        V_t = self.ocv - np.multiply(I_t, Ro) - np.multiply(I_R1, R1) - np.multiply(I_R2, R2)
        assert V_t.shape == self.voltage.shape
        data_len = V_data.shape[0]
        MAPE = (np.sum(np.abs(V_t - V_data) / V_data)) / data_len
        MSE = np.sqrt(np.sum(np.square(V_t - V_data)) / data_len)
        max_APE = (np.max(np.abs(V_t - V_data) / V_data))
        if save:
            print("Saving error statistics...")
            np.savetxt('../batt_sys_identification/{}_MAPE_{}_{}.csv'.format(trial, cell_name, diagn), [MAPE])
            np.savetxt('../batt_sys_identification/{}_MSE_{}_{}.csv'.format(trial, cell_name, diagn), [MSE])
            np.savetxt('../batt_sys_identification/{}_max_APE_{}_{}.csv'.format(trial, cell_name, diagn), [max_APE])
            print('Done saving error statistics.')
        return MSE, MAPE*100, max_APE*100

    def _validate_params(self):
        """
        Internal method.
        Validates that the parameter list has been updated, will throw an exception if list is not updated.

        :raise : Assertion error if not validated.
        :return: None.
        """
        assert len(self.params) > 0

    def plot_correction_scheme_comparison(self, xlim=(20000, 36000), ylim=(2.75, 3.85)):
        """
        Generates a plot of OCV corrected model and non-OCV corrected model.

        :return: None
        """
        v_corr = self.get_corrected_voltages()
        v_uncorr = self.get_uncorrected_voltages()
        plt.figure()
        plt.plot(self.voltage, label='experimental data')
        plt.plot(v_uncorr, '--', label='ECM without OCV correction')
        plt.plot(v_corr, '--', label='ECM with OCV correction', color='k')
        # plt.xlim(xlim)
        # plt.ylim(ylim)
        plt.xticks(rotation=45)
        plt.xlabel("Time Step (s)")
        plt.ylabel("Voltage")
        plt.legend()
        plt.tight_layout()
        plt.savefig('../batt_sys_identification/model_comparison_ocv_correction_scheme.png')

    def get_uncorrected_voltages(self):
        """
        Returns the vector of uncorrected voltages from ECM model response.

        :return: Vector of uncorrected voltages.
        """
        A_Ro = self.params_uncorr[0]
        B_Ro = self.params_uncorr[1]
        C_Ro = self.params_uncorr[2]
        R1 = self.params_uncorr[3]
        C1 = self.params_uncorr[4]
        R2 = self.params_uncorr[5]
        C2 = self.params_uncorr[6]
        dt = 1
        Ro = B_Ro * np.exp(C_Ro * self.soc) + A_Ro * np.exp(self.soc)  # updated the new Ro based on intuition
        I_t = -self.current  # this will be loaded from .npy
        I_R1 = np.zeros((I_t.shape[0],))
        I_R2 = np.zeros((I_t.shape[0],))

        for j in range(1, len(I_R1)):
            I_R1[j] = np.round(np.exp(-dt / (R1 * C1)) * I_R1[j - 1] + (1 - np.exp(-dt / (R1 * C1))) * I_t[j - 1], 10)
            I_R2[j] = np.round(np.exp(-dt / (R2 * C2)) * I_R2[j - 1] + (1 - np.exp(-dt / (R2 * C2))) * I_t[j - 1], 10)
        V_t = self.ocv_no_corr - np.multiply(I_t, Ro) - np.multiply(I_R1, R1) - np.multiply(I_R2, R2)
        return V_t

    def get_corrected_voltages(self):
        """
        Returns the voltage response with corrected open circuit voltage.

        :return: Vector of voltage response with corrected open circuit voltage.
        """
        A_Ro = self.params[0]
        B_Ro = self.params[1]
        C_Ro = self.params[2]
        R1 = self.params[3]
        C1 = self.params[4]
        R2 = self.params[5]
        C2 = self.params[6]
        dt = 1

        Ro = B_Ro * np.exp(C_Ro * self.soc) + A_Ro * np.exp(self.soc)  # updated the new Ro based on intuition
        I_t = -self.current  # this will be loaded from .npy
        I_R1 = np.zeros((I_t.shape[0],))
        I_R2 = np.zeros((I_t.shape[0],))

        for j in range(1, len(I_R1)):
            I_R1[j] = np.round(np.exp(-dt / (R1 * C1)) * I_R1[j - 1] + (1 - np.exp(-dt / (R1 * C1))) * I_t[j - 1], 10)
            I_R2[j] = np.round(np.exp(-dt / (R2 * C2)) * I_R2[j - 1] + (1 - np.exp(-dt / (R2 * C2))) * I_t[j - 1], 10)
        V_t = self.ocv - np.multiply(I_t, Ro) - np.multiply(I_R1, R1) - np.multiply(I_R2, R2)
        return V_t

    def get_Ro(self):
        """
        Returns the high frequency (Ro) resistance of the battery.

        :return: Resistance (R_o).
        """
        A_Ro = self.params[0]
        B_Ro = self.params[1]
        C_Ro = self.params[2]
        return B_Ro * np.exp(C_Ro * self.soc) + A_Ro * np.exp(self.soc)

    def plot_Ro(self, grid=False):
        """
        Plots the high frequency resistance (Ro) of the battery.

        :return: None.
        """
        Ro = self.get_Ro()

        # Create a figure and axis with custom size
        fig, ax = plt.subplots()

        # Plot Ro vs. SoC with markers and line
        ax.plot(self.soc, Ro, linestyle='-', linewidth=4, label='Internal Resistance (Ro)')

        # Add labels and title with increased font size
        ax.set_xlabel('State of Charge (SoC)', fontsize=14)
        ax.set_ylabel('Internal Resistance $R_o$ (Ohm)', fontsize=14)
        ax.set_title('$R_o$ vs. State of Charge', fontsize=14)

        # Add a grid for better readability (optional)
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('../batt_sys_identification/Ro_vs_SoC.png', dpi=300)

        plt.show()

    def simulate_response(self):
        """
        Simulates the response of the ECM model. Not complete yet.

        :return:
        """
        # voltage = self.get_uncorrected_voltages()
        return NotImplementedError("Method has not been implemented yet!")

    def run_pre_checks(self):
        """
        Runs pre-checks on input values to ensure they are all of desired lengths. This is run at the beginning to
        ensure no errors in data length.

        :return: None.
        """
        self._assert_data_len()

    def _estimate_soc_vector(self):
        """
        Estimates the SOC vector for users that do not have the vector but have the capacity and
        current profiles. Not fully implemented yet.

        :return: Error message.
        """
        return NotImplementedError("Method has not been implemented yet!")


def run_trials_paper(path):
    """
    Runs the system identification algorithm for a number of trials and saves the results to file. This runs the trials
    used in the battery system identification portion of the paper.

    :return: None.
    """
    NUM_TRIALS = 10
    for i in range(NUM_TRIALS):
        print(f'Running trial {i+1} of {NUM_TRIALS}...')
        battery_data = pd.read_csv(path)
        m = BatteryParams(battery_data)
        m.run_sys_identification(use_initial_pop=False, cell_name=path.split('_')[4], diagn=i + 1, save=True)
        m.plot_correction_scheme_comparison()
        print(f'Done with trial {i+1} of {NUM_TRIALS}...')
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
    mape_list = [np.loadtxt(f'0_MAPE_{cell_name}_{i+1}.csv')*100 for i in range(num_trials)]
    mse_list = [np.loadtxt(f'0_MSE_{cell_name}_{i+1}.csv')*100 for i in range(num_trials)]
    max_ape_list = [np.loadtxt(f'0_max_APE_{cell_name}_{i+1}.csv')*100 for i in range(num_trials)]

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


def create_error_boxplots():
    """
    Creates the box plots used in the original paper. This is for the error statistics and non-functional
    to the main functionality of the package.

    :return: None.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming you have error data for W8, W9, and W10
    NUM_TRIALS = 10
    error_data_w8, _, _ = make_error_vector('W8', NUM_TRIALS)
    error_data_w9, _, _  = make_error_vector('W9', NUM_TRIALS)
    error_data_w10, _, _ = make_error_vector('W10', NUM_TRIALS)

    # Combine the data into a list
    error_data = [error_data_w8, error_data_w9, error_data_w10]

    # Create a box plot
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

    # Add a grid for better readability
    # ax.yaxis.grid(True, linestyle='--', alpha=0.7)

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


if __name__ == '__main__':
    # These are for testing purposes only.
    # create_error_boxplots()
    # create_ro_plot()
    import os
    # import multiprocessing as mp
    # # List all the csv files in current directory and let user choose one.
    # # User uploaded data will be saved in the current directory as a temp_data.csv file.
    # # data_path = os.path.join(os.getcwd(), 'temp_data.csv')
    # data_paths = [
    #     'batt_iden_test_data_W8_1.csv',
    #     'batt_iden_test_data_W9_1.csv',
    #     'batt_iden_test_data_W10_1.csv'
    # ]
    # use_cores_count = min(mp.cpu_count(), len(data_paths))
    #
    # with mp.get_context("spawn").Pool(use_cores_count) as pool:
    #     print(f"Running {use_cores_count} parallel battery trial processes")
    #     pool.map(run_trials_paper, data_paths)

    # NUM_TRIALS = 10

    data_path = 'batt_sys_identification/batt_iden_test_data_W8_1.csv'  # Change this to the path of your data file.
    batt_data = pd.read_csv(data_path)
    module = BatteryParams(batt_data)
    # Toggle initial population on or off. Set to ``False`` to toggle off.
    module.run_sys_identification(use_initial_pop=True)
    module.plot_correction_scheme_comparison()
