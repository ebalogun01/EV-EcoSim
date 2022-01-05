import cvxpy as cp


class Optimization:
    """This is a class for the optimization problem we will solve. It contains all necessary attributes
    that fully define the optimization problem."""
    def __init__(self, objective_type, objective, battery_constraint, power_demand, time_res, transformer, battery, time=0, name=None):
        self._objective_type = objective_type
        self._objective = objective
        self._name = name
        self._time = time
        self._constraints = []
        self.test_demand = power_demand
        self.time_res = time_res
        self.charge = 0
        self.discharge = 0
        self._transformer = transformer
        self._battery = battery
        self.problem = None
        self.market_constraints = None
        self.battery_constraints = battery_constraint

    def get_battery_constraint(self):
        return self.battery_constraints

    def get_market_constraints(self):
        return self.market_constraints

    def aggregate_constraints(self):
        if self.battery_constraints:
            self._constraints.extend(self.battery_constraints)
        if self.market_constraints:
            self._constraints.extend(self.market_constraints)

    def get_constraints(self):
        return self._constraints

    def run(self):
        """runs an instance of the problem"""
        self.aggregate_constraints()    # aggregate constraints
        problem = cp.Problem(cp.Minimize(self._objective), self._constraints)
        self.problem = problem
        result = problem.solve(verbose=False)
        print(problem.status)

    @staticmethod
    def get_final_states(battery):
        print("SOC battery: ", battery.getSOC())
        return battery.getSOC()









