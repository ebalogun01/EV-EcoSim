from config import Config
import json


class SimRun:
    # Class initialization with default values
    def __init__(self):
        self.run_period_name = "Not-initialized"
        self.config = Config()

    def save_config_to_json(self):
        with open('input/sim_run_settings.json', 'w') as settings_json:
            json.dump(self.config.get_config_json(), settings_json, indent=1)
