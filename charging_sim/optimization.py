"""Contains the Optimization class, which is used by controller to solve the optimization problem."""
import cvxpy as cp


class Optimization:
    """
    Constructor for the overall optimization problem solved by the optimization-based controller.
    * Designed to include future cost functions such as transformer degradation and battery aging.
    * Limited to convex and Mixed Integer programs, depending on the selected solver.
    * Note, each desired solver must be installed separately on user's PC for a successful run.

    :param objective_type: Type of objective problem being optimizaed.
    :param objective: CVXPY objective function object.
    :param controller: Controller object.
    :param power_demand: Power demand at the Charging Station.
    :param time_res: Time resolution of problem data.
    :param transformer: Transformer object (optional, not implemented yet).
    :param battery: Battery object.
    :param time: Time Counter
    :param name: Optimization identifier.
    :param solar: Solar Object
    :param str solver: Available backend solver to invoke (ECOS, MOSEK, GUROBI, etc.).

    """

    # TODO: change all refs to battery_constraints to call controller
    def __init__(self, objective_type, objective, controller, power_demand, time_res, transformer, battery, time=0,
                 name=None, solar=None, solver='GUROBI'):

        self._objective_type = objective_type
        self._objective = objective
        self._name = name
        self._time = time
        self._constraints = []
        self.test_demand = power_demand
        self.time_res = time_res
        self.charge = 0
        self.discharge = 0
        self.transformer = transformer
        self.battery = battery
        self.solar = solar
        self.controller = controller  # update this somewhere else in simulation
        self.problem = None
        self.market_constraints = None
        self.battery_constraints = controller.get_battery_constraints(power_demand)
        self.cost_per_opt = []
        self.solver = getattr(cp, solver)
        if solar:
            setattr(self, 'solar', solar)

    def build_emissions_cost(self):
        """
        Builds emission cost to be included in the objective function (future work).

        :return:
        """
        pass

    def build_battery_cost(self):
        """
        Build battery cost (to be implemented in future version).

        :return:
        """
        pass

    def build_transformer_cost(self):
        """
        Build Transformer cost (to be implemented in future version).

        :return:
        """
        pass

    def add_demand_charge(self, charge):
        """
        Including demand charge in the objective function (Deprecated).

        :param charge: Demand charge ($/kW).
        :return:
        """
        load = 0    # placeholder
        cost = charge * load + (self.controller.battery_power_charge +
                                self.controller.battery_power_discharge -
                                self.solar.battery_power - self.solar.ev_power)

    def get_battery_constraint(self):
        """
        Returns the list of battery constraints within the controller.

        :return: Battery constraints.
        """
        return self.battery_constraints

    def aggregate_constraints(self):
        """
        Aggregates all the constraints into one constraint list within the object.

        :return: None.
        """
        if self.battery_constraints:  # fix this later to call battery directly
            self._constraints.extend(self.battery_constraints)
        if self.market_constraints:
            self._constraints.extend(self.market_constraints)
        if self.solar:
            self._constraints.extend(self.solar.get_constraints())

    def get_constraints(self):
        """
        Returns the constraints list.

        :return: List of all constraints within the problem.
        """
        return self._constraints

    def run(self):
        """
        Runs an instance of the optimization problem.

        :return float: Optimal objective value.
        """
        self.aggregate_constraints()  # aggregate constraints
        problem = cp.Problem(cp.Minimize(self._objective), self._constraints)
        self.problem = problem
        result = problem.solve(solver=self.solver, verbose=False)
        self.cost_per_opt.append(result)
        # print(problem.status) ACTIVATE LATER
        return result
