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
        lb2 = [-0.1, -0.1, -5, 0.0001, 100, 0.0004, 4000]
        ub2 = [0.1, 0.1, 0, 0.02, 10000, 0.01, 10000]

        lb3 = [-10, -10, -10, 0.00004, 0.03, 0.0004, 600]  # change the range for testing capacitance variation with SOC if any
        ub3 = [10, 10, 10, 0.01, 0.06, 0.01, 10000]

        lb = [0.02, 0.0001, 100, 0.00004, 900]
        ub = [0.04, 0.01, 10000, 0.004, 10000]
        gene_space_range = [{'low': low, 'high': high} for low, high in zip(lb2, ub2)]
        return gene_space_range

    def ga(self, num_generations=5, num_parents_mating=2, sol_per_pop=10, num_genes=7, crossover_type="single_point",
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
                                   parallel_processing=8)
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
                                   parallel_processing=8)

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        ga_instance.plot_fitness()
        print(f'Parameters of the best solution : {solution}')
        print(f"Fitness value of the best solution = {solution_fitness}")
        return solution

    def run_sys_identification(self, cell_name=0, diagn=0, use_initial_pop=True, quadratic_bias=True):
        """
        Runs the GA for system identification.

        :return:
        """
        if use_initial_pop:
            self._init_population()
        self.params = self.ga()

        if quadratic_bias:
            self.run_ocv_correction(use_quadratic=True)
        else:
            self.run_ocv_correction()   # Uses simple linear bias correction.
        # Run again.
        self.params = self.ga()     # New params.
        np.savetxt(f'battery_params_{cell_name}_{diagn}.csv', self.params, delimiter=',')

    def run_ocv_correction(self, use_quadratic=False, cell_name=0, diagn=0):
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
        self.data.to_csv('../batt_sys_identification/input_data_with_ocv_corr_voltage.csv')
        #    Adds corrected ocv as field and writes the input data

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

    def plot_Ro(self):
        """
        Plots the high frequency resistance (Ro) of the battery.

        :return: None.
        """
        Ro = self.get_Ro() + self.params[3] + self.params[5]
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.soc, Ro)
        plt.title("Ro vs. SoC plot")
        plt.xlabel('State of charge')
        plt.ylabel('Ro (Resistance)')
        plt.show()

    def simulate_response(self):
        """
        Simulates the response of the ECM model. Not complete yet.

        :return:
        """
        voltage = self.get_uncorrected_voltages()
        # todo: complete later

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


if __name__ == '__main__':
    import os
    # List all the csv files in current directory and let user choose one.
    # User uploaded data will be saved in the current directory as a temp_data.csv file.
    data_path = os.path.join(os.getcwd(), 'temp.csv')
    batt_data = pd.read_csv(data_path)
    module = BatteryParams(batt_data)
    # Create button option to toggle initial population on or off.
    module.run_sys_identification(use_initial_pop=True)
    module.plot_correction_scheme_comparison()
