import unittest
import sys
sys.path.append('../..')
from batt_sys_identification.battery_identification import BatteryParams
import pandas as pd


data = pd.read_csv('../../batt_sys_identification/paper_data/batt_iden_test_data_W10_1.csv')

class MyTestCase(unittest.TestCase):

    """
    Test module for battery system identification module.
    Must pass before any changes are pushed to main.
    """

    def test_initialization(self):
        module = BatteryParams(data)
        self.assertTrue(module is not None)

    def test_ga(self):
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
