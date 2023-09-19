TEXT = {
    'intro': 'We have made EV-Ecosim a simple, online platform tailored for scientific and non-technical communities immersed in Electric Vehicle (EV) infrastructure planning and operation. Its robust co-simulation framework empowers researchers, experts, practitioners, or any curious person to explore the design of efficient, sustainable EV charging infrastructures harmonized with DERs, fostering deeper understanding in this evolving domain.',
    'howToUse': 'This tool can be leveraged through a simple 3-step process: Setup-Simulate-Assess. First the simulation parameters need to be defined, where either a prepared scenario can be run, or the input may be fully customized. Then, the defined scenario is run. As this is a complex computation, this step may take significant time periods. After the output is calculated, a dashboard is shown with visualised key data, and the data may be exported for further analysis, or the simulation may be run again with adjusted input.',
    'credits1': 'This online tool was developed by Emmanuel Balogun, Marek Miltner and Koye Alagbe of Stanford University to accompany the publication \"EV-ecosim: A grid-aware co-simulation platform for the design and optimization of electric vehicle charging stations\".',
    'credits2': 'If you use this tool or its source code, you are required to cite the original paper accessible via the preprint button at the top right of this page.',
    'credits3': 'We also recommend citing GridLAB-D as it is an internal core this work is building on.',
    'ambientDataTooltip': 'Only the time fields, temperature field, and year are needed, please provide at 15min intervals',
    'solarDataTooltip': 'Only Month, Day, Hour, and GHI needed. Please provide data at 15 minutes intervals for a year',
    'batteryDataTooltip': 'Required fields are indicated in the data prototype at a 1s resolution',
    'priceDataTooltip': 'Include the electricity time-of-use rates for a whole year, at 15 min intervals',
    'oneShotTooltip': 'This mode runs the full optimization for an entire horizon (default is a month) and leverages '
                      'the solution within the simulation. This will run faster than an MPC mode, which solves and optimization problem'
                      'at each time step or interval.',
    'feederPopTooltip': 'This is the feeder population module within the battery test case. This file performs the pre-simulation step for '
                        'running EV-Ecosim. It takes in a base Gridlab-D Model (GLM) file (for example, `IEEE123.glm`), and modifies that file by including'

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

MONTH_DROPDOWN = [
    {'label': 'January', 'value': '1'},
    {'label': 'February', 'value': '2'},
    {'label': 'March', 'value': '3'},
    {'label': 'April', 'value': '4'},
    {'label': 'May', 'value': '5'},
    {'label': 'June', 'value': '6'},
    {'label': 'July', 'value': '7'},
    {'label': 'August', 'value': '8'},
    {'label': 'September', 'value': '9'},
    {'label': 'October', 'value': '10'},
    {'label': 'November', 'value': '11'},
    {'label': 'December', 'value': '12'}
]