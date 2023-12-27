import unittest
import sys
sys.path.append('../..')
from charging_sim.clock import Clock
import json


class MyTestCase(unittest.TestCase):
    """
    Basic test for clock class.

    """
    def test_initialization(self):
        """
        Simple basic test to ensure that the clock module can be initialized to a valid object. This test will fail if
        the module cannot be initialized.

        :return: None.
        """
        # Load the JSON config file.
        with open('../../charging_sim/configs/clock.json', 'r') as f:
            config = json.load(f)
        module = Clock(config)
        print(module)
        self.assertTrue(module is not None)
        print("Clock basic test passed!")


if __name__ == '__main__':
    unittest.main()
