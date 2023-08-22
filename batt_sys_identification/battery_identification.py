import numpy as np
import matplotlib.pyplot as plt
import copy
import pygad
import cvxpy as cp
import time


class BatteryParams:
    """
    Battery system identification class with open circuit voltage correction scheme.
    Takes in dataframe or csv with some given fields during instantiation.

    {
    Dataframe fields (columnsmust include:
    * current
    * voltage
    }


     To use:
    * data = pd.read_csv(data_path). This loads a pandas dataframe.
    * module = BatteryParams(data)
    * module.run_sys_identification()
    * module.plot_correction_scheme_comparison()

    :param data: Battery test data from which to fit the identification params.
    """
    def __init__(self, data):
        self.data = data
        self.current = data.current.values
        self.voltage = data.voltage.values
        self.ocv = data.ocv.values
        self.ocv_no_corr = data.ocv.values     # uncorrected open circuit voltage
        self.soc = data.soc.values
        self.params = []
        self.params_uncorr = []

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

    @staticmethod
    def _get_pygad_bounds():
        """
        Returns gene space range, representing the min and max values for each gene in the parameter vector.

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

    def ga(self, num_generations=10, num_parents_mating=2, sol_per_pop=10, num_genes=7, crossover_type="single_point",
           mutation_type="adaptive", parent_selection_type="sss", mutation_percent_genes=60,
           mutation_prob=(0.3, 0.1)):
        """
        Runs the genetic algorithm instance. Please see PyGAD documentation for more explanation of fields/params.
        The default parameters have been selected to optimize accuracy and speed, however, any user may find a
        combination of params that work better for a given set of battery data.

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
        gene_space_range = self._get_pygad_bounds()
        # define instance of GA
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=self._cost_fun_pygad,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               mutation_probability=mutation_prob,
                               parent_selection_type=parent_selection_type,
                               crossover_type=crossover_type,
                               allow_duplicate_genes=False,
                               stop_criteria=["reach_0.05", "saturate_20"],
                               gene_space=gene_space_range)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        ga_instance.plot_fitness()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        return solution

    def run_sys_identification(self):
        """
        Runs the GA for system identification
        :return:
        """
        self.params = self.ga()
        self.run_ocv_correction()
        # run again
        self.params = self.ga()     # new params

    def run_ocv_correction(self, use_quadratic=True, cell_name=0, diagn=0):
        """
        This fits the parameters for the open circuit voltage correction scheme. Updates the ocv attribute.

        :return: None.
        """
        self.validate_params()
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
        np.savetxt('OCV_bias_correction_params_{}_{}.csv'.format(cell_name, diagn), ocv_bias_correction_vector)
        self.ocv = ocv_corr.value

    def validate_params(self):
        """
        Validates that the parameter list has been updated.
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
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(rotation=45)
        plt.xlabel("Time Step (s)")
        plt.ylabel("Voltage")
        plt.legend()
        plt.tight_layout()
        plt.savefig('model_comparison_ocv_correction_scheme.png')

    def get_uncorrected_voltages(self):
        """
        Returns the vector of uncorrected voltages from ECM model response.

        :return:
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

        :return:
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

    def run_pre_checks(self):
        """
        Runs pre-checks on input values to ensure they are all of desired lengths. This is run at the beginning to
        ensure no errors in data length.

        :return: None.
        """
        self._assert_data_len()

    def _estimate_soc_vector(self):
        """
        This method helps estimate the SOC vector for users that do not have the vector but have the capacity and
        current profiles.
        :return:
        """
        return NotImplementedError("Method has not been implemented yet!")



