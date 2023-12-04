import os

from config import Config
import json


class SimRun:
    # Class initialization with default values
    def __init__(self, config):
        self.run_period_name = "Not-initialized"
        self.config = config

    def save_config_to_json(self):
        print(os.getcwd())
        with open('input/sim_run_settings.json', 'w') as settings_json:
            json.dump(self.config, settings_json, indent=1)
