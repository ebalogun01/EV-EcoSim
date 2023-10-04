import os.path
import sys

sys.path.append('../charging_sim')
sys.path.append('../analysis')

import plotly.express as px
import dash
import pandas as pd
from dash import dcc, html, ctx, Input, Output, State, no_update
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from components import create_home_page, create_tutorial_page, create_output_page
from config import Config
from sim_run import SimRun
from constants import PRESET as preset
from run_simulation import *
import base64
import datetime
import io
from analysis import load_post_opt_costs as post_opt_module
import time

#   Create Dash app
app = dash.Dash(__name__)

# Create config object
user_input = Config()  # Default setup

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
    html.Div(className="background",
             children=[
                 html.Div(className="content",
                          children=[
                              dmc.LoadingOverlay(
                                  html.Div(
                                      className="content-container",
                                      id="main-container",
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
                                  loaderProps={"variant": "oval", "color": "red", "size": "xl"},
                              ),
                          ]),
             ]),
])


# Callbacks

# Loading state TODO end when function is done
@app.callback(
    Output("main-container", "children"),
    Input("run-simulation-button", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    """
    This function causes a delay when the run simulation button is clicked before loading the results

    :param n_clicks: the number of clicks of the Run simulation button (Unused)

    :returns: dash.no_update to prevent main-container from updating
    """
    time.sleep(2)
    return no_update


# Radio buttons change value
@app.callback(
    Output(component_id="home-page", component_property="style"),
    Output(component_id="tutorial-page", component_property="style"),
    Output(component_id="output-page", component_property="style"),
    Input(component_id="category-radio", component_property="value")
)
def page_update(radio_value):
    """
    This function updates the visible page according to the radio buttons (To be deprecated by pagination)

    :param radio_value: The selected page

    :return: Style objects for the pages to be rendered
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
        return {'display': 'none'}, {'display': 'none'}


# Preset selected
@app.callback(
    Output(component_id="preset-button", component_property="className"),
    Output(component_id="custom-settings-button", component_property="className"),
    Output(component_id="custom-settings-accordion", component_property="value"),
    Output(component_id="run-simulation-button", component_property="className"),
    Output(component_id="run-simulation-button", component_property="disabled"),
    Input(component_id="preset-button", component_property="n_clicks"),
    Input(component_id="custom-settings-button", component_property="n_clicks"),
    Input(component_id="custom-settings-accordion", component_property="value"),
    State(component_id="preset-button", component_property="className"),
    State(component_id="custom-settings-button", component_property="className"),
    prevent_initial_call=True
)
def select(preset_n_clicks, custom_settings_n_clicks, custom_settings_value,
           preset_class, custom_settings_class):
    """
    This function updates the preset, custom settings, and run simulation buttons visually when either the preset or custom settings is
    selected. Can be triggered by the preset button, the custom settings button, or the custom settings accordion.

    :param preset_n_clicks: Number of clicks of the preset button (Unused)
    :param custom_settings_n_clicks: Number of clicks of the custom settings button (Unused)
    :param custom_settings_value: Value of the custom settings accordion indicating whether it was opened or closed
    :param preset_class: The class of the preset button before the trigger
    :param custom_settings_class: The class of the custom settings button before the trigger

    :return: Updated classes for the buttons (selected/not selected) and value for the accordion element
    """
    triggered_id = ctx.triggered_id
    print(triggered_id)
    if triggered_id == "preset-button":
        # Load preset
        return "setup-button selected tooltip", "setup-button tooltip", None, "action tooltip", False
    elif triggered_id == "custom-settings-button":
        return "setup-button tooltip", "setup-button selected tooltip", "customSettings", "action tooltip", False
    elif triggered_id == "custom-settings-accordion":
        # Load custom settings
        print(custom_settings_value)
        if custom_settings_value == "customSettings":
            return "setup-button tooltip", "setup-button selected tooltip", "customSettings", "action tooltip", False
        else:
            return preset_class, custom_settings_class, None, "action tooltip", False
    else:
        return "setup-button tooltip", "setup-button tooltip", None,  "action disabled tooltip", True


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
def select(oneshot_n_clicks, mpc_rhc_n_clicks, battery_system_n_clicks):
    """
    Updates the classes of the simulation mode buttons when a simulation mode is selected

    :param oneshot_n_clicks: Number of clicks of the oneshot button (unused)
    :param mpc_rhc_n_clicks: Number of clicks of the MPC/RHC button (unused)
    :param battery_system_n_clicks: Number of clicks of the battery system identification button (unused)

    :return: Updated classes for the buttons (selected/not selected) and visibility for their corresponding input fields
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
    Updates the temperature data field with the uploaded file name

    :param contents: Uploaded file contents
    :param name: Name of uploaded file

    :return: Uploaded file name if something was uploaded
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
    Updates the solar data field with the uploaded file name

    :param contents: Uploaded file contents
    :param name: Name of uploaded file

    :return: Uploaded file name if something was uploaded
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
    Updates the load data field with the uploaded file name

    :param contents: Uploaded file contents
    :param name: Name of uploaded file

    :return: Uploaded file name if something was uploaded
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
    Updates the price data field with the uploaded file name

    :param contents: Uploaded file contents
    :param name: Name of uploaded file

    :return: Uploaded file name if something was uploaded
    """
    if contents is not None:
        return name
    else:
        return "No file chosen"


# Battery data uploaded
@app.callback(
    Output(component_id="battery-data-file", component_property="children"),
    Output(component_id="run-battery-system-identification-button", component_property="className"),
    Output(component_id="run-battery-system-identification-button", component_property="disabled"),
    Input(component_id="battery-data-upload", component_property="contents"),
    State(component_id="battery-data-upload", component_property="filename")
)
def battery_upload(contents, name):
    """
    Updates the battery data field with the uploaded file name

    :param contents: Uploaded file contents
    :param name: Name of uploaded file

    :return: Uploaded file name if something was uploaded
    """
    if contents is not None:
        df = parse_contents_to_df(contents, name)
        df.to_csv('../batt_sys_identification/temp.csv', index=False)
        return name, 'action tooltip', False
    else:
        return "No file chosen", 'action disabled tooltip', True


# Feeder population data uploaded
@app.callback(
    Output(component_id="feeder-population-data-file", component_property="children"),
    Input(component_id="feeder-population-data-upload", component_property="contents"),
    State(component_id="feeder-population-data-upload", component_property="filename")
)
def feeder_population_upload(contents, name):
    """
    Updates the feeder population data field with the uploaded file name

    :param contents: Uploaded file contents
    :param name: Name of uploaded file

    :return: Uploaded file name if something was uploaded
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
    Updates the power factor label when the slider is adjusted

    :param value: Slider value

    :return: The value to populate the label with
    """
    return value


# Run simulation
@app.callback(
    Output(component_id="run-simulation-button", component_property="style"),
    Input(component_id="run-simulation-button", component_property='n_clicks'),
    State(component_id="preset-button", component_property="className"),
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
        preset_class,
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
    Collects all the simulation settings from the settings inputs and sends them to the backend to run the simulation.

    :return: Style object for the run simulation button (Only used because Dash requires that callbacks have an output)
    """
    # either use preset, or user_input depending on which is selected
    user_input = Config()
    if preset_class == "setup-button selected tooltip":
        user_input = preset
    elif custom_settings_class == "setup-button selected tooltip":
        if oneshot_class == "setup-button selected":
            user_input.sim_mode = "offline"
        elif mpc_rhc_class == "setup-button selected":
            user_input.sim_mode = "mpc_rhc"
        elif battery_system_class == "setup-button selected":
            user_input.sim_mode = "battery"
            user_input.only_batt_sys = True  # I think?
        if battery_system_class == "setup-button selected":
            # user_input.battery["data"] = battery_filename
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
            user_input.charging_station["commercial_building_trans"] = float(
                transformer_capactiy)  # is this the correct property? Yes -Emmanuel

            # Decomposed this code block.
            max_c_rate = aggregate_user_battery_inputs(max_c_rate_1, max_c_rate_2, max_c_rate_3, max_c_rate_4,
                                                       max_c_rate_5)
            energy_cap = aggregate_user_battery_inputs(energy_cap_1, energy_cap_2, energy_cap_3, energy_cap_4,
                                                       energy_cap_5)
            max_ah = aggregate_user_battery_inputs(max_ah_1, max_ah_2, max_ah_3, max_ah_4, max_ah_5)
            max_voltage = aggregate_user_battery_inputs(max_voltage_1, max_voltage_2, max_voltage_3, max_voltage_4,
                                                        max_voltage_5)

            user_input.battery["max_c_rate"] = max_c_rate
            user_input.battery["pack_energy_cap"] = energy_cap
            user_input.battery["pack_max_Ah"] = max_ah  # Would make an exhaustive list later
            user_input.battery["pack_max_voltage"] = max_voltage
            user_input.battery["power_factor"] = float(power_factor)
            # Nothing for capacity
            if mpc_rhc_class == "setup-button selected":
                user_input.feeder_pop = True
                # Nothing for feeder pop data

    # Get json from config
    user_input_dict = user_input.get_config_json()
    # user_input_dict = user_input    # Already a default Dict.

    sim_run = SimRun(user_input_dict)

    # Save input as json
    sim_run.save_config_to_json()

    # Connect to backend here - pass user_input.get_config_json()
    print('Simulation start...')
    print(type(user_input_dict))
    print(user_input_dict)
    simulate(user_input_dict)
    print("Simulation complete!")
    return {'grid-row': '2'}


@app.callback(
    Output(component_id="run-battery-system-identification-button", component_property="style"),
    Input(component_id="run-battery-system-identification-button", component_property="n_clicks"),
    prevent_initial_call=True
)
def run_battery_system_identification(run_battery_system_n_clicks):
    """
    Run battery simulation identification module

    :param run_battery_system_n_clicks: Number of clicks on the run battery system identification button (unused)

    :return: Battery System identifiaction data file settings
    """
    # TODO: Include option to choose number of populations for the GA algorithm.
    print("batt sys trigger")
    from batt_sys_identification.battery_identification import BatteryParams
    try:
        battery_data = pd.read_csv('../batt_sys_identification/temp.csv')
        module = BatteryParams(battery_data)
        module.run_sys_identification()

    except Exception as e:
        print(e)
        return html.Div(['No file uploaded for battery system identification!'])

    print("Run Battery System Identification done!")
    return {'position': 'relative', 'float': 'right'}


@app.callback(
    Output(component_id="run-post-opt-analysis-button", component_property="style"),
    Input(component_id="run-post-opt-analysis-button", component_property="n_clicks"),
    prevent_initial_call=True
)
def run_post_opt_analysis(run_post_opt_analysis_n_clicks):
    """
    Runs the post-simulation cost analysis and produces the relevant files and results.

    :param run_post_opt_analysis_n_clicks:
    :return:
    """
    # first check if the results folder exists
    if not os.path.isdir('../analysis/results'):
        raise FileNotFoundError('result/ folder does not exist, please run simulation first!')

    post_opt_module.run()   # Runs the post-optimization cost module.
    print("Run Post Sim Analysis done!")
    return {'grid-row': '2'}


def aggregate_user_battery_inputs(input_1, input_2, input_3, input_4, input_5):
    """
    Aggregate battery inputs from front end

    :return: list of inputs as floats
    """
    input_list = []
    if input_1:
        input_list.append(float(input_1))
    if input_2:
        input_list.append(float(input_2))
    if input_3:
        input_list.append(float(input_3))
    if input_4:
        input_list.append(float(input_4))
    if input_5:
        input_list.append(float(input_5))

    return input_list


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
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df

# Max Voltage-Max Ah dependency for all inputs
@app.callback(
    Output(component_id='max-ah-input1', component_property='value'),
    Input(component_id='energy-cap-input1', component_property='value'),
    Input(component_id='max-voltage-input1', component_property='value'),
)
def populate_max_ah_input1(energy_cap, voltage):
    """
        Updates the value of the max Ah input depending on the first energy capacity and voltage

        :param energy_cap: The energy capacity as a string
        :param voltage: The voltage as a string

        :return: The max Ah value
    """
    if energy_cap == '' or voltage == '':
        return ''
    else:
        return float(energy_cap) / float(voltage)
    
@app.callback(
    Output(component_id='max-ah-input2', component_property='value'),
    Input(component_id='energy-cap-input2', component_property='value'),
    Input(component_id='max-voltage-input2', component_property='value'),
)
def populate_max_ah_input2(energy_cap, voltage):
    """
    Updates the value of the max Ah input depending on the second energy capacity and voltage

    :param energy_cap: The energy capacity as a string
    :param voltage: The voltage as a string

    :return: The max Ah value
    """
    if energy_cap == '' or voltage == '':
        return ''
    else:
        return float(energy_cap) / float(voltage)
    
@app.callback(
    Output(component_id='max-ah-input3', component_property='value'),
    Input(component_id='energy-cap-input3', component_property='value'),
    Input(component_id='max-voltage-input3', component_property='value'),
)
def populate_max_ah_input3(energy_cap, voltage):
    """
    Updates the value of the max Ah input depending on the third energy capacity and voltage

    :param energy_cap: The energy capacity as a string
    :param voltage: The voltage as a string

    :return: The max Ah value
    """
    if energy_cap == '' or voltage == '':
        return ''
    else:
        return float(energy_cap) / float(voltage)

@app.callback(
    Output(component_id='max-ah-input4', component_property='value'),
    Input(component_id='energy-cap-input4', component_property='value'),
    Input(component_id='max-voltage-input4', component_property='value'),
)
def populate_max_ah_input4(energy_cap, voltage):
    """
    Updates the value of the max Ah input depending on the fourth energy capacity and voltage

    :param energy_cap: The energy capacity as a string
    :param voltage: The voltage as a string

    :return: The max Ah value
    """
    if energy_cap == '' or voltage == '':
        return ''
    else:
        return float(energy_cap) / float(voltage)
    
@app.callback(
    Output(component_id='max-ah-input5', component_property='value'),
    Input(component_id='energy-cap-input5', component_property='value'),
    Input(component_id='max-voltage-input5', component_property='value'),
)
def populate_max_ah_input5(energy_cap, voltage):
    """
    Updates the value of the max Ah input depending on the fifth energy capacity and voltage

    :param energy_cap: The energy capacity as a string
    :param voltage: The voltage as a string

    :return: The max Ah value
    """
    if energy_cap == '' or voltage == '':
        return ''
    else:
        return float(energy_cap) / float(voltage)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
