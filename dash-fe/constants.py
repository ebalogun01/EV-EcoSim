TEXT = {
    'intro': 'EV-Ecosim is a sophisticated online platform tailored for scientific and technical communities immersed in electric vehicle (EV) charging and distributed energy resources (DERs). By delving into the intricate dynamics of EV charging configurations, the tool reveals teh interplay between battery capacity, cost optimization, and grid interactions. Its robust co-simulation framework empowers researchers and experts to explore the design of efficient, sustainable EV charging infrastructures harmonized with DERs, fostering deeper understanding in this evolving domain.',
    'howToUse': 'This tool can be leveraged through a simple 3-step process: Setup-Simulate-Assess. First the simulation parameters need to be defined, where either a prepared scenario can be run, or the input may be fully customized. Then, the defined scenario is run. As this is a complex computation, this step may take significant time periods. After the output is calculated, a dashboard is shown with visualised key data, and the data may be exported for further analysis, or the simulation may be run again with adjusted input.',
    'credits1': 'This online tool was created by Emmanuel Balogun, Lily Buechler, Marek Miltner and Koye Alagbe of Stanford University to accompany the publication \"EV-ecosim: A grid-aware co-simulation platform for the design and optimization of electric vehicle charging stations\".',
    'credits2': 'If you use this tool or the corresponding source code of EV-Ecosim for a publication you are required to cite it, e.g., Balogun, E., Buechler, L., Miltner, M., Alagbe, K., \"EV-Ecosim Version major.minor.patch-build (branch) platform\", (year) [online]. Available at url, Accessed on: month day, year.',
    'credits3': 'We also recommend citing GridLAB-D as it is an internal core this work is building on.',
}

#TODO: UPDATE PRESETS IF NEEDED
PRESET1 = {
    "sim_mode": "offline",
    "feeder_pop": False,
    "only_batt_sys": False,
    "ambient_data": False,
     "month": 7,
    "num_days": 30,
    "solar": {
        "data": None,
        "efficiency": None,
        "rating": None
    },
    "battery": {
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
    },
    "charging_station": {
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
    },
    "load": {
        "data": None
    },
    "elec_prices": {
        "data": None,
        "month": 7
    }
}

PRESET2 = {
    "sim_mode": "offline",
    "feeder_pop": False,
    "only_batt_sys": False,
    "ambient_data": False,
     "month": 7,
    "num_days": 30,
    "solar": {
        "data": None,
        "efficiency": None,
        "rating": None
    },
    "battery": {
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
    },
    "charging_station": {
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
    },
    "load": {
        "data": None
    },
    "elec_prices": {
        "data": None,
        "month": 7
    }
}