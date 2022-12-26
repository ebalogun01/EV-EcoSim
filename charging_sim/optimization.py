import cvxpy as cp


class Optimization:
    """This is a class for the optimization problem we will solve. It contains all necessary attributes
    that fully define the optimization problem."""

    # TODO: change all refs to battery_constraints to call controller
    def __init__(self, objective_type, objective, controller, power_demand, time_res, transformer, battery, time=0,
                 name=None, solar=None, solver='MOSEK'):
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

    def objective(self):
        pass

    def build_emissions_cost(self):
        pass

    def build_battery_cost(self):
        pass

    def build_transformer_cost(self):
        pass

    def add_demand_charge(self, charge):
        """Optimizing for demand charge by simply minimizing each daily max
        MIGHT NOT USE THIS FOR NOW"""
        cost = charge * load + (self.controller.battery_power_charge +
                                self.controller.battery_power_discharge -
                                self.solar.battery_power - self.solar.ev_power)

    def build_electricity_cost(self, load, energy_prices_TOU, demand_charge=False):
        # to be used later. For now, keep as-is.
        """Need to update from home load right now; maybe this can be useful in future opt."""
        # TODO: include time-shifting for energy TOU price rates? Add emissions cost pricing based on TOD?
        lam = 10  # this needs to be guided
        sparsity_cost_factor = 0.0  # dynamically determine this in future based on load * cost
        sparsity_cost = cp.norm(self.controller.battery_power_charge, 1) + \
                        cp.norm(self.controller.battery_power_discharge, 1)
        cost_electricity = cp.sum((cp.multiply(energy_prices_TOU, (load + (self.controller.battery_power_charge +
                                                                           self.controller.battery_power_discharge -
                                                                           self.solar.battery_power -
                                                                           self.solar.ev_power))))) + \
                           sparsity_cost_factor * sparsity_cost
        if demand_charge:
            demand_charge_cost = cp.max(cp.pos(load + (self.controller.battery_power_charge +
                                                       self.controller.battery_power_discharge -
                                                       self.solar.battery_power - self.solar.ev_power)))
            cost_electricity += demand_charge_cost
        return cost_electricity

    def get_battery_constraint(self):
        return self.battery_constraints

    def get_market_constraints(self):
        return self.market_constraints

    def aggregate_constraints(self):
        if self.battery_constraints:  # fix this later to call battery directly
            self._constraints.extend(self.battery_constraints)
        if self.market_constraints:
            self._constraints.extend(self.market_constraints)
        if self.solar:
            self._constraints.extend(self.solar.get_constraints())

    def get_constraints(self):
        return self._constraints

    def run(self):
        """runs an instance of the problem
        Using ecos as default solver as it gives the lowest optimal value"""
        self.aggregate_constraints()  # aggregate co----nstraints
        problem = cp.Problem(cp.Minimize(self._objective), self._constraints)
        self.problem = problem
        result = problem.solve(solver=self.solver, verbose=True)
        self.cost_per_opt.append(result)
        # print(problem.status) ACTIVATE LATER
        return result

    @staticmethod
    def get_final_states(battery):
        print("SOC battery: ", battery.getSOC())
        return battery.getSOC()


def test():
    """This is solely used for testing"""
    pass


if __name__ == "__main__":
    test()
