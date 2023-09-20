from config import Config

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

preset_config = Config()
preset_config.sim_mode = "offline"
preset_config.feeder_pop = False
preset_config.only_batt_sys = False
preset_config.ambient_data = False
preset_config.month = 7
preset_config.num_days = 30

preset_config.solar["data"] = None
preset_config.solar["efficiency"] = None
preset_config.solar["rating"] = None

preset_config.battery["data"] = None
preset_config.battery["max_c_rate"] = [1, 2],
preset_config.battery["pack_max_voltage"] = [250, 400],
preset_config.battery["pack_energy_cap"] = [int(5e4), int(10e4)],
preset_config.battery["pack_max_Ah"] = [250, 400],
preset_config.battery["SOC_min"] = 0.2
preset_config.battery["SOC_max"] = 0.9
preset_config.battery["params_0_cycles"] = "/battery_data/params_OCV_corr_W8_1"
preset_config.battery["OCV_map_SOC"] = "/battery_data/SOC_corr_W8_1.csv"
preset_config.battery["OCV_map_voltage"] == "/battery_data/OCV_corr_W8_1.csv"

preset_config.charging_station["power_factor"] = None
preset_config.charging_station["dcfc_power_cap"] = 700
preset_config.charging_station["power_cap_units"] = "kW"
preset_config.charging_station["dcfc_charging_stall_base_rating"] = "75_kW"
preset_config.charging_station["l2_charging_stall_base_rating"] = "11.5_kW"
preset_config.charging_station["num_dcfc_nodes"] = 1
preset_config.charging_station["num_l2_nodes"] = 0
preset_config.charging_station["num_dcfc_stalls_per_node"] = 5
preset_config.charging_station["num_l2_stalls_per_node"] = 0
preset_config.charging_station["commercial_building_trans"] = 75

preset_config.load["data"] = None
preset_config.elec_prices["data"] = None
preset_config.elec_prices["month"] = 7

PRESET2 = preset_config

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
