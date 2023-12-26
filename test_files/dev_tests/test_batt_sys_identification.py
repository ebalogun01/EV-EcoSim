import unittest
import sys
sys.path.append('../..')
from batt_sys_identification.battery_identification import BatteryParams
import pandas as pd


data = pd.read_csv('../../batt_sys_identification/data/batt_iden_test_data_W10_1.csv')

class MyTestCase(unittest.TestCase):

    """
    Test module for battery system identification module.
    Must pass before any changes are pushed to main.
    """

    def test_initialization(self):
        """
        Tests that the battery system identification module can be initialized to a valid object.

        :return: None.
        """
        module = BatteryParams(data)
        self.assertTrue(module is not None)

    def test_ga(self):
        """
        This runs the battery system identification module with the genetic algorithm. It ensures that it runs
        to completion and that the output is the correct size. Using an assertion to check that the output is the
        sufficient and efficient to check that the output is correct without knowing the exact values of the output.
        This test will fail if entire module method does not run to completion.

        :return: None.
        """
        num_seq_runs = 1
        module = BatteryParams(data)
        for i in range(num_seq_runs):
            module.run_sys_identification(use_initial_pop=False, generations=2)
            # module.plot_correction_scheme_comparison()
            self.assertEqual(len(module.params), 7)
            self.assertEqual(len(module.params_uncorr), 7)
            # module.plot_Ro()


if __name__ == '__main__':
    unittest.main()
