import unittest
from charging_sim.clock import Clock
import json


class MyTestCase(unittest.TestCase):
    """
    Basic test for clock class.

    """
    def test_initialization(self):
        # Load the JSON config file.
        with open('../../charging_sim/configs/clock.json', 'r') as f:
            config = json.load(f)
        module = Clock(config)
        print(module)
        self.assertTrue(module is not None)
        print("Clock basic test passed!")


if __name__ == '__main__':
    unittest.main()
