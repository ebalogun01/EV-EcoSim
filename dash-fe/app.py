import sys
sys.path.append('../charging_sim')
import plotly.express as px
import dash
import pandas as pd
from dash import dcc, html, ctx, Input, Output, State
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from components import create_home_page, create_tutorial_page, create_output_page
from config import Config
from sim_run import Sim_run
from constants import PRESET1, PRESET2
from run_simulation import *
import base64
import datetime
import io



#   Create Dash app
app = dash.Dash(__name__)

# Create app layout
app.layout = html.Div([
    html.Link(rel="stylesheet", href="assets/style.css"),

    # Header
    html.Div([
        dmc.Header(
            height=70,
            fixed=True,
            pl=0,
            pr=0,
            pt=0,
            style={'background-color': '#fdd', 'color': 'whitesmoke'},
            children=[
                dmc.Group(
                    position="apart",
                    align="center",
                    children=[
                        dcc.Link(
                            dmc.ThemeIcon(
                                html.Img(src='assets/doerr.png', width="150"),
                                style={"paddingLeft": 225, "paddingTop": 28},
                                variant="transparent",
                            ),
                            href='https://sustainability.stanford.edu/',
                            target='_blank'
                        ),
                        dcc.Link(
                            href=app.get_relative_path("/"),
                            style={"paddingTop": 15, "paddingLeft": 150, "paddingBottom": 5, "paddingRight": 10,
                                   "textDecoration": "none"},
                            children=[
                                dmc.Group(align='center',
                                          spacing=0,
                                          position='center',
                                          children=[
                                              dmc.Text("EV-Ecosim",
                                                       color="dark",
                                                       style={'font-family': 'Arial', 'fontSize': 24},
                                                       weight=700),
                                          ]
                                          )
                            ]
                        ),
                        dmc.Group(
                            position="right",
                            align="center",
                            style={"paddingRight": 155, "paddingTop": 8},
                            children=[
                                dmc.Group(
                                    position="right",
                                    align="center",
                                    children=[
                                        html.A(
                                            dmc.Badge(
                                                "Source code",
                                                leftSection=dmc.ThemeIcon(
                                                    DashIconify(icon='mdi:github'),
                                                    color='dark',

                                                ),
                                                sx={"paddingLeft": 5},
                                                size="lg",
                                                radius="lg",
                                                variant="filled",
                                                color="dark"
                                            ),
                                            href='https://github.com/ebalogun01/EV50_cosimulation',
                                            target='_blank'
                                        )
                                    ],
                                ),
                                dmc.Group(
                                    position="right",
                                    align="center",
                                    children=[
                                        html.A(
                                            dmc.Badge(
                                                "Documentation",
                                                leftSection=dmc.ThemeIcon(
                                                    DashIconify(icon='mdi:book-alphabet'),
                                                    color='dark'
                                                ),
                                                sx={"paddingLeft": 5},
                                                size="lg",
                                                radius="lg",
                                                variant="filled",
                                                color="dark"
                                            ),
                                            href="https://ebalogun01.github.io/EV50_cosimulation/",
                                            target='_blank'
                                        )
                                    ],
                                ),
                                dmc.Group(
                                    position="right",
                                    align="right",
                                    children=[
                                        html.A(
                                            dmc.Badge(
                                                "Preprint",
                                                leftSection=dmc.ThemeIcon(
                                                    DashIconify(icon='mdi:file-document'),
                                                    color='dark',
                                                ),
                                                sx={"paddingLeft": 5},
                                                size="lg",
                                                radius="lg",
                                                variant="filled",
                                                color="dark"
                                            ),
                                            href='https://www.techrxiv.org/articles/preprint/EV-ecosim_A_grid-aware_co-simulation_platform_for_the_design_and_optimization_of_electric_vehicle_charging_stations/23596725',
                                            target='_blank'
                                        )
                                    ],
                                )
                            ],
                        )
                    ]
                ),
            ]
        ),
    ]),

    # Body
    html.Div(className="background", children=[
        html.Div(className="content", children=[
            html.Div(
                className="content-container",
                style={'margin-top': '100px'},
                children=[
                    html.Div([
                        dcc.RadioItems(
                            id="category-radio",
                            value="INP",
                            options=[
                                {"label": "Input", "value": "INP"},
                                {"label": "Tutorial", "value": "TUT"},
                                {"label": "Output", "value": "OUT"}],
                            inline=True,
                        ),
                        html.Br(),
                    ]),

                    create_home_page(),
                    create_tutorial_page(),
                    create_output_page(),
                ],
            ),
        ]),
    ]),
])

# Callbacks
# Radio buttons change value
@app.callback(
    Output(component_id="home-page", component_property="style"),
    Output(component_id="tutorial-page", component_property="style"),
    Output(component_id="output-page", component_property="style"),
    Input(component_id="category-radio", component_property="value")
)
def page_update(radio_value):
    """
    Updating page based on selected radio button (To be deprecated by pagination)

    :return: Page to be rendered
    """
    # Debug option:
    if app.server.debug == True:
        pass
        print(radio_value)
    if radio_value == "INP":
        return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
    elif radio_value == "TUT":
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    elif radio_value == "OUT":
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display':'none'}
    
# Preset selected
@app.callback(
    Output(component_id="preset1-button", component_property="className"),
    Output(component_id="preset2-button", component_property="className"),
    Output(component_id="custom-settings-button", component_property="className"),
    Output(component_id="custom-settings-accordion", component_property="value"),
    Input(component_id="preset1-button", component_property="n_clicks"),
    Input(component_id="preset2-button", component_property="n_clicks"),
    Input(component_id="custom-settings-button", component_property="n_clicks"),
    Input(component_id="custom-settings-accordion", component_property="value"),
    State(component_id="preset1-button", component_property="className"),
    State(component_id="preset2-button", component_property="className"),
    State(component_id="custom-settings-button", component_property="className"),
    prevent_initial_call=True
)
def select(preset1_n_clicks, preset2_n_clicks, custom_settings_n_clicks, custom_settings_value, preset1_class, preset2_class, custom_settings_class):
    """
    Preset selection trigger

    :return: Preset
    """
    triggered_id = ctx.triggered_id
    print(triggered_id)
    if triggered_id == "preset1-button":
        # Load preset 1
        return "setup-button selected tooltip", "setup-button tooltip", "setup-button tooltip", None
    elif triggered_id == "preset2-button":
        # Load preset 2
        return "setup-button tooltip", "setup-button selected tooltip", "setup-button tooltip", None
    elif triggered_id == "custom-settings-button":
        return "setup-button tooltip", "setup-button tooltip", "setup-button selected tooltip", "customSettings"
    elif triggered_id == "custom-settings-accordion":
        # Load custom settings
        print(custom_settings_value)
        if custom_settings_value == "customSettings":
            return "setup-button tooltip", "setup-button tooltip", "setup-button selected tooltip", "customSettings"
        else:
            return preset1_class, preset2_class, custom_settings_class, None
    else:
        return "setup-button tooltip", "setup-button tooltip", "setup-button tooltip", None
    
# Simulation mode selected
@app.callback(
    Output(component_id="oneshot-button", component_property="className"),
    Output(component_id="mpc-rhc-button", component_property="className"),
    Output(component_id="battery-system-button", component_property="className"),
    Output(component_id="simulation-container", component_property="style"),
    Output(component_id="feeder-population-container", component_property="style"),
    Output(component_id="battery-system-container", component_property="style"),
    Input(component_id="oneshot-button", component_property="n_clicks"),
    Input(component_id="mpc-rhc-button", component_property="n_clicks"),
    Input(component_id="battery-system-button", component_property="n_clicks"),
    prevent_initial_call=True
)
def select(oneshot_n_clicks, mpc_rhc_n_clicks, battery_systes_n_clicks):
    """
    Sim mode selection trigger

    :return: Sim mode
    """
    triggered_id = ctx.triggered_id
    print(triggered_id)
    if triggered_id == "oneshot-button":
        # Hide Feeder Population
        return "setup-button selected", "setup-button", "setup-button", {
            'display': 'grid',
            'grid-row': '4',
            'grid-column': '1 / span 6',
            'grid-column-gap': '20px',
            'grid-row-gap': '20px',
            'grid-template-rows': 'repeat(20, auto)',
            'grid-template-columns': 'repeat(6, minmax(0, 1fr))'
        }, {
            'display': 'none',
        }, {
            'display': 'none',
        }
    elif triggered_id == "mpc-rhc-button":
        # Show everything
        return "setup-button", "setup-button selected", "setup-button", {
            'display': 'grid',
            'grid-row': '4',
            'grid-column': '1 / span 6',
            'grid-column-gap': '20px',
            'grid-row-gap': '20px',
            'grid-template-rows': 'repeat(20, auto)',
            'grid-template-columns': 'repeat(6, minmax(0, 1fr))'
        }, {
            'display': 'grid',
            'grid-row-gap': '20px',
            'grid-row': '7 / span 2',
            'grid-column': '2 / span 4'
        }, {
            'display': 'none',
        }
    elif triggered_id == "battery-system-button":
        # Load custom settings
        return "setup-button", "setup-button", "setup-button selected", {
            'display': 'none',
        }, {
            'display': 'none',
        }, {
            'display': 'block',
            'grid-row': '4',
            'grid-column': '2 / span 4',
        }
    else:
        return "setup-button", "setup-button", "setup-button", {
            'display': 'grid',
            'grid-row': '3',
            'grid-column': '1 / span 6',
            'grid-column-gap': '20px',
            'grid-row-gap': '20px',
            'grid-template-rows': 'repeat(20, auto)',
            'grid-template-columns': 'repeat(6, minmax(0, 1fr))'
        }, {
            'display': 'none',
        }, {
            'display': 'none',
        },

# Temperature data uploaded
@app.callback(
    Output(component_id="temperature-data-file", component_property="children"),
    Input(component_id="temperature-data-upload", component_property="contents"),
    State(component_id="temperature-data-upload", component_property="filename")
)
def temperature_upload(contents, name):
    """
    Temperature upload trigger

    :return: File for temperature
    """
    if contents is not None:
        return name
    else:
        return "No file chosen"
    
# Solar data uploaded
@app.callback(
    Output(component_id="solar-data-file", component_property="children"),
    Input(component_id="solar-data-upload", component_property="contents"),
    State(component_id="solar-data-upload", component_property="filename")
)
def solar_upload(contents, name):
    """
    Solar upload trigger

    :return: File for solar
    """
    if contents is not None:
        return name
    else:
        return "No file chosen"
    
# Load data uploaded
@app.callback(
    Output(component_id="load-data-file", component_property="children"),
    Input(component_id="load-data-upload", component_property="contents"),
    State(component_id="load-data-upload", component_property="filename")
)
def load_upload(contents, name):
    """
    Load upload trigger

    :return: File for load
    """
    if contents is not None:
        return name
    else:
        return "No file chosen"
    
# Price data uploaded
@app.callback(
    Output(component_id="price-data-file", component_property="children"),
    Input(component_id="price-data-upload", component_property="contents"),
    State(component_id="price-data-upload", component_property="filename")
)
def price_upload(contents, name):
    """
    Price upload trigger

    :return: File for price
    """
    if contents is not None:
        return name
    else:
        return "No file chosen"
    
# Battery data uploaded
@app.callback(
    Output(component_id="battery-data-file", component_property="children"),
    Output(component_id="run-battery-system-identification-button", component_property="className"),
    Input(component_id="battery-data-upload", component_property="contents"),
    State(component_id="battery-data-upload", component_property="filename")
)
def battery_upload(contents, name):
    """
    Battery upload trigger

    :return: File for battery
    """
    if contents is not None:
        df = parse_contents_to_df(contents, name)
        df.to_csv('../batt_sys_identification/temp.csv', index=False)
        return name, 'action tooltip'
    else:
        return "No file chosen", 'action disabled tooltip'
    
# Feeder population data uploaded
@app.callback(
    Output(component_id="feeder-population-data-file", component_property="children"),
    Input(component_id="feeder-population-data-upload", component_property="contents"),
    State(component_id="feeder-population-data-upload", component_property="filename")
)
def feeder_population_upload(contents, name):
    """
    Feeder popúulation upload trigger

    :return: File for feeder popúulation
    """
    if contents is not None:
        # todo: save the feeder_pop file in the required folder
        return name
    else:
        return "No file chosen"
    
# Power factor adjusted
@app.callback(
    Output(component_id="power-factor-label", component_property="children"),
    Input(component_id="power-factor-slider", component_property="value")
)
def power_factor_update(value):
    """
    Power factor trigger

    :return: Power factor
    """
    return value

# Run simulation
@app.callback(
    Output(component_id="run-simulation-button", component_property="style"),
    Input(component_id="run-simulation-button", component_property='n_clicks'),
    State(component_id="preset1-button", component_property="className"),
    State(component_id="preset2-button", component_property="className"),
    State(component_id="custom-settings-button", component_property="className"),
    State(component_id="oneshot-button", component_property="className"),
    State(component_id="mpc-rhc-button", component_property="className"),
    State(component_id="battery-system-button", component_property="className"),
    State(component_id="battery-data-upload", component_property="filename"),
    State(component_id="temperature-data-upload", component_property="filename"),
    State(component_id="load-data-upload", component_property="filename"),
    State(component_id="solar-data-upload", component_property="filename"),
    State(component_id="solar-efficiency-input", component_property="value"),
    State(component_id="solar-capacity-input", component_property="value"),
    State(component_id="month-dropdown", component_property="value"),
    State(component_id="year-input", component_property="value"),
    State(component_id="days-input", component_property="value"),
    State(component_id="price-data-upload", component_property="filename"),
    State(component_id="dcfc-rating-dropdown", component_property="value"),
    State(component_id="l2-rating-dropdown", component_property="value"),
    State(component_id="num-dcfc-stalls-dropdown", component_property="value"),
    State(component_id="num-l2-stalls-dropdown", component_property="value"),
    State(component_id="transformer-capacity-dropdown", component_property="value"),
    State(component_id="max-c-rate-input1", component_property="value"),
    State(component_id="max-c-rate-input2", component_property="value"),
    State(component_id="max-c-rate-input3", component_property="value"),
    State(component_id="max-c-rate-input4", component_property="value"),
    State(component_id="max-c-rate-input5", component_property="value"),
    State(component_id="energy-cap-input1", component_property="value"),
    State(component_id="energy-cap-input2", component_property="value"),
    State(component_id="energy-cap-input3", component_property="value"),
    State(component_id="energy-cap-input4", component_property="value"),
    State(component_id="energy-cap-input5", component_property="value"),
    State(component_id="max-ah-input1", component_property="value"),
    State(component_id="max-ah-input2", component_property="value"),
    State(component_id="max-ah-input3", component_property="value"),
    State(component_id="max-ah-input4", component_property="value"),
    State(component_id="max-ah-input5", component_property="value"),
    State(component_id="max-voltage-input1", component_property="value"),
    State(component_id="max-voltage-input2", component_property="value"),
    State(component_id="max-voltage-input3", component_property="value"),
    State(component_id="max-voltage-input4", component_property="value"),
    State(component_id="max-voltage-input5", component_property="value"),
    State(component_id="power-factor-slider", component_property="value"),
    State(component_id="battery-capacity-input", component_property="value"),
    State(component_id="feeder-population-data-upload", component_property="filename"),
    prevent_initial_call=True
)
def run_simulation(
        run_button_n_clicks,
        preset1_class, 
        preset2_class, 
        custom_settings_class,
        oneshot_class,
        mpc_rhc_class,
        battery_system_class,
        battery_filename,
        temperature_filename,
        load_filename,
        solar_filename,
        solar_efficiency,
        solar_capacity,
        month,
        year,
        num_days,
        price_filename,
        dcfc_rating,
        l2_rating,
        num_dcfc_stalls,
        num_l2_stalls,
        transformer_capactiy,
        max_c_rate_1,
        max_c_rate_2,
        max_c_rate_3,
        max_c_rate_4,
        max_c_rate_5,
        energy_cap_1,
        energy_cap_2,
        energy_cap_3,
        energy_cap_4,
        energy_cap_5,
        max_ah_1,
        max_ah_2,
        max_ah_3,
        max_ah_4,
        max_ah_5,
        max_voltage_1,
        max_voltage_2,
        max_voltage_3,
        max_voltage_4,
        max_voltage_5,
        power_factor,
        battery_capacity,
        feeder_population_filename
    ):
    """
    Simulation settings collection trigger

    :return: Simulation settings JSON
    """

    # either use preset_1, preset_2, or user_input depending on which is selected
    user_input = Config() # Default setup
    if preset2_class == "setup-button selected tooltip":
        user_input = PRESET2 # TODO: Create config object from preset 2
    elif custom_settings_class == "setup-button selected tooltip":
        user_input = Config()
        if oneshot_class == "setup-button selected":
            user_input.sim_mode = "offline"
        elif mpc_rhc_class == "setup-button selected":
            user_input.sim_mode = "mpc_rhc"
        elif battery_system_class == "setup-button selected":
            user_input.sim_mode = "battery"
            user_input.only_batt_sys = True # I think?
        if battery_system_class == "setup-button selected":
            #user_input.battery["data"] = battery_filename
            return {'grid-row': '2'}
        else:
            user_input.ambient_data = temperature_filename
            user_input.load["data"] = load_filename
            user_input.solar["data"] = solar_filename
            user_input.solar["efficiency"] = float(solar_efficiency)
            user_input.solar["rating"] = float(solar_capacity)
            user_input.month = int(month)
            # No provision for year
            user_input.num_days = int(num_days)
            user_input.elec_prices["data"] = price_filename
            user_input.charging_station["dcfc_charging_stall_base_rating"] = dcfc_rating + "_kW"
            user_input.charging_station["l2_charging_stall_base_rating"] = l2_rating + "_kW"
            user_input.charging_station["num_dcfc_stalls_per_node"] = int(num_dcfc_stalls)
            user_input.charging_station["num_l2_stalls_per_node"] = int(num_l2_stalls)
            user_input.charging_station["commercial_building_trans"] = float(transformer_capactiy) # is this the correct property? Yes -Emmanuel
            # TODO: fill in bettery dropdown values and format the values accordingly
            max_c_rate = []
            energy_cap = []
            max_ah = []
            max_voltage = []

            if max_c_rate_1 and max_c_rate_1 != '':
                max_c_rate.append(float(max_c_rate_1))
            if max_c_rate_2 and max_c_rate_2 != '':
                max_c_rate.append(float(max_c_rate_2))
            if max_c_rate_3 and max_c_rate_3 != '':
                max_c_rate.append(float(max_c_rate_3))
            if max_c_rate_4 and max_c_rate_4 != '':
                max_c_rate.append(float(max_c_rate_4))
            if max_c_rate_5 and max_c_rate_5 != '':
                max_c_rate.append(float(max_c_rate_5))

            if energy_cap_1 and energy_cap_1 != '':
                energy_cap.append(float(energy_cap_1))
            if energy_cap_2 and energy_cap_2 != '':
                energy_cap.append(float(energy_cap_2))
            if energy_cap_3 and energy_cap_3 != '':
                energy_cap.append(float(energy_cap_3))
            if energy_cap_4 and energy_cap_4 != '':
                energy_cap.append(float(energy_cap_4))
            if energy_cap_5 and energy_cap_5 != '':
                energy_cap.append(float(energy_cap_5))

            if max_ah_1 and max_ah_1 != '':
                max_ah.append(float(max_ah_1))
            if max_ah_2 and max_ah_2 != '':
                max_ah.append(float(max_ah_2))
            if max_ah_3 and max_ah_3 != '':
                max_ah.append(float(max_ah_3))
            if max_ah_4 and max_ah_4 != '':
                max_ah.append(float(max_ah_4))
            if max_ah_5 and max_ah_5 != '':
                max_ah.append(float(max_ah_5))

            if max_voltage_1 and max_voltage_1 != '':
                max_voltage.append(float(max_voltage_1))
            if max_voltage_2 and max_voltage_2 != '':
                max_voltage.append(float(max_voltage_2))
            if max_voltage_3 and max_voltage_3 != '':
                max_voltage.append(float(max_voltage_3))
            if max_voltage_4 and max_voltage_4 != '':
                max_voltage.append(float(max_voltage_4))
            if max_voltage_5 and max_voltage_5 != '':
                max_voltage.append(float(max_voltage_5))

            print(max_c_rate, energy_cap, max_ah, max_voltage)
            user_input.battery["max_c_rate"] = max_c_rate 
            user_input.battery["pack_energy_cap"] = energy_cap
            user_input.battery["pack_max_Ah"] = max_ah  # Would make an exhaustive list later
            user_input.battery["pack_max_voltage"] = max_voltage
            user_input.battery["power_factor"] = float(power_factor)
            # Nothing for capacity
            if mpc_rhc_class == "setup-button selected":
                user_input.feeder_pop = True
                # Nothing for feeder pop data
        user_input = user_input.get_config_json()

    # Connect to backend here - pass user_input.get_config_json()
    simulate(user_input)

    return {'grid-row': '2'}

@app.callback(
    Output(component_id="run-battery-system-identification-button", component_property="style"),
    Input(component_id="run-battery-system-identification-button", component_property="n_clicks")
)
def run_battery_system_identification(run_battery_system_n_clicks):
    # TODO: Include option to choose number of populations for the GA algorithm.
    from batt_sys_identification.battery_identification import BatteryParams
    try:
        battery_data = pd.read_csv('../batt_sys_identification/temp.csv')
        module = BatteryParams(battery_data)
        module.run_sys_identification()

    except Exception as e:
        print(e)
        return html.Div(['No file uploaded for battery system identification!'])

    print("Run Battery System done")
    return {'position': 'relative', 'float': 'right'}


def parse_contents_to_df(contents, filename):
    """
    This function takes in the string bytes of uploaded csv and excel file and returns a Pandas df.
    Throws an exception and returns an error message if file is not csv or excel spreadsheet.
    Code reference: https://dash.plotly.com/dash-core-components/upload

    :param contents: Contents from the str bytes parsed from upload.
    :param filename: Name of file uploaded.
    :return: Pandas dataframe of contents.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = None
    # todo: need to fix input data to not include index column else this can be buggy in the future.
    #  Leaving for now for testing purposes.
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), index_col=0)
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded), index_col=0)
            print(df)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
