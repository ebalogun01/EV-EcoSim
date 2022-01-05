import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Solar:
    """This simulates the ground-truth Solar conditions in location"""
    config: dict
    data_path: str

    def __post_init__(self):
        self.solar_df = pd.read_csv(self.config["data_path"]).to_numpy()
        self.solar_vec = self.solar_df.to_numpy()
        self.location = self.config["location"]
        self.start_year = self.config["start_year"]
        self.start_month = self.config["start_month"]
        self.start_day = self.config["start_day"]
        self.efficiency = self.config["efficiency"]
        self.resolution = self.config["resolution"]
        self.sample_start_idx = self.config["sample_start_idx"]
        self.sample_end_idx = self.config["sample_end_idx"]

    def get_power(self):
        return self.solar_vec[self.sample_start_idx:self.sample_end_idx]

