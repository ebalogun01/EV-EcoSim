import unittest
import sys
sys.path.append('../..')    # For Ubuntu compatibility.
from charging_sim.transformer import OilTypeTransformer as Transformer
from charging_sim.clock import Clock
import pandas as pd
import json

# Load transformer config JSON file.
with open('../../charging_sim/configs/transformer.json', 'r') as f:
    config = json.load(f)

# Load the clock config file.
with open('../../charging_sim/configs/clock.json', 'r') as f:
    clock_config = json.load(f)


class MyTestCase(unittest.TestCase):

    """
    Test module for battery system identification module.
    Must pass before any changes are pushed to main.
    """

    def test_initialization(self):
        """
        Tests that the transformer module can be initialized to a valid object.

        :return: None.
        """
        module = Transformer(config)
        self.assertTrue(module is not None)

    def test_dynamics(self):
        """
        Tests the thermal dynamics of the transformer module. This test will fail if the transformer thermal dynamics
        do not run to completion. It also checks that the states of the transformer are updated correctly and are of the
        consistent length. This does not check the accuracy of the state propagation model, but a basic sanity check is
        performed to ensure that the transformer is at least at ambient temperature (steady state), else the dynamics
        cannot be correct.

        :return: None
        """
        # Read transformer test load data from data/trans_test_load.csv
        trans_load = pd.read_csv('../data/trans_test_load.csv')['station_net_grid_load_kW']

        # Load the clock config file.

        global_clock = Clock(clock_config)
        module = Transformer(config, global_clock=global_clock)
        for power in trans_load:
            module.thermal_dynamics(power)
            global_clock.update()   # Update the clock.
        self.assertTrue(len(module.To_list) >= len(trans_load))     # Testing To state update.
        self.assertTrue(len(module.Th_list) >= len(trans_load))     # Testing Th state update.
        self.assertTrue(min(module.To_list) >= module.Ta)   # Ensuring transformer is at least ambient (steady state).
        self.assertTrue(len(module.To_list) == len(module.Th_list))  # Ensuring To and Th are same length.
        for i in range(len(module.To_list)):
            self.assertTrue(module.To_list[i] <= module.Th_list[i])  # Ensuring hotspot is at least top oil temp (steady state).

        print("Transformer basic test passed!")
        print("Plotting transformer thermal states...")
        import matplotlib.pyplot as plt
        plt.plot(module.To_list, label='Top oil temperature')
        plt.plot(module.Th_list, label='Hot-spot temperature')
        plt.plot(module.Ta_list, label='Ambient temperature')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
