# simulation config setup

"""
Sample structure:

{
    "sim_mode": "offline",
    "feeder_pop": false,
    "only_batt_sys": false,
    "ambient_data": null,
    "month": 7      ,
    "num_days": 30,
    "solar": {
        "data": null,
        "efficiency": null,
        "rating": null
    },
    "battery": {
        "data": null,
        "max_c_rate": [1, 2],
        "pack_max_voltage": [250, 400],
        "pack_energy_cap": [5e4, 10e4],
        "pack_max_Ah": [250, 400],
        "SOC_min": 0.2,
        "SOC_max": 0.9,
        "params_0_cycles": "/battery_data/params_OCV_corr_W8_1",
        "OCV_map_SOC": "/battery_data/SOC_corr_W8_1.csv",
        "OCV_map_voltage": "/battery_data/OCV_corr_W8_1.csv"
    },
    "charging_station": {
        "power_factor": null,
        "dcfc_power_cap": 700,
        "power_cap_units": "kW",
        "dcfc_charging_stall_base_rating": "75_kW",
        "l2_charging_stall_base_rating": "11.5_kW",
        "num_dcfc_nodes": 1,
        "num_l2_nodes": 0,
        "num_dcfc_stalls_per_node": 5,
        "num_l2_stalls_per_node": 0,
        "commercial_building_trans": 75
    },
    "load": {
        "data": null
    },
    "elec_prices": {
        "data": null,
        "month": 7
    }

}

"""


class Config:
    # Class initialiyation with default values
    def __init__(self):
        self.sim_mode = "offline"
        self.feeder_pop = False
        self.only_batt_sys = False
        self.ambient_data = None
        self.month = 7
        self.num_days = 30
        self.solar = {
            "data": None,
            "efficiency": None,
            "rating": None
        }
        self.battery = {
            "data": None,
            "max_c_rate": [1, 2],
            "pack_max_voltage": [250, 400],
            "pack_energy_cap": [5e4, 10e4],
            "pack_max_Ah": [250, 400],
            "SOC_min": 0.2,
            "SOC_max": 0.9,
            "params_0_cycles": "/battery_data/params_OCV_corr_W8_1",
            "OCV_map_SOC": "/battery_data/SOC_corr_W8_1.csv",
            "OCV_map_voltage": "/battery_data/OCV_corr_W8_1.csv"
        }
        self.charging_station = {
            "power_factor": None,
            "dcfc_power_cap": 700,
            "power_cap_units": "kW",
            "dcfc_charging_stall_base_rating": "75_kW",
            "l2_charging_stall_base_rating": "11.5_kW",
            "num_dcfc_nodes": 1,
            "num_l2_nodes": 0,
            "num_dcfc_stalls_per_node": 5,
            "num_l2_stalls_per_node": 0,
            "commercial_building_trans": 75
        }
        self.load = {
            "data": None
        }
        self.elec_prices = {
            "data": None,
            "month": 7
        }

    # Prints JSON for debug purposes
    def __str__(self):
        print("JSON TBD")
        return (self.get_config_json())

    # TODO Generates JSON
    def get_config_json(self):
        return ""