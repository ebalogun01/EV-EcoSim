import cvxpy as cp


class Optimization:
    """This is a class for the optimization problem we will solve. It contains all necessary attributes
    that fully define the optimization problem."""
    def __init__(self, objective_type, objective, constraint, power_demand, time_res, transformer, battery, time=0, name=None):
        self._objective_type = objective_type
        self._objective = objective
        self._name = name
        self._time = time
        self._constraints = constraint
        self.test_demand = power_demand
        self.time_res = time_res
        self.charge = 0
        self.discharge = 0
        self._transformer = transformer
        self._battery = battery
        self.problem = None

    def run(self):
        """runs an instance of the problem"""
        problem = cp.Problem(cp.Minimize(self._objective), self._constraints)
        self.problem = problem
        result = problem.solve(verbose=False)
        print(problem.status)

    @staticmethod
    def get_final_states(battery):
        print(battery.getSOC())
        return battery.getSOC()









