import plotly.express as px
import dash
from dash import dcc, html, ctx, Input, Output, State
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from components import create_home_page, create_tutorial_page, create_output_page
from config import Config
from constants import PRESET1, PRESET2

# Create Dash app
app = dash.Dash(__name__)

# Create default config file
cfg = Config()
print(cfg)

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
    # Debug option:
    if app.server.debug==True:
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
    Output(component_id="mode-helper-text", component_property="children"),
    Input(component_id="oneshot-button", component_property="n_clicks"),
    Input(component_id="mpc-rhc-button", component_property="n_clicks"),
    Input(component_id="battery-system-button", component_property="n_clicks"),
    prevent_initial_call=True
)
def select(oneshot_n_clicks, mpc_rhc_n_clicks, battery_systes_n_clicks):
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
        }, 'Helper text 1'
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
        }, 'Helper text 2'
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
        }, 'Helper text 3'
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
    if contents is not None:
        return name
    else:
        return "No file chosen"
    
# Battery data uploaded
@app.callback(
    Output(component_id="battery-data-file", component_property="children"),
    Input(component_id="battery-data-upload", component_property="contents"),
    State(component_id="battery-data-upload", component_property="filename")
)
def battery_upload(contents, name):
    if contents is not None:
        return name
    else:
        return "No file chosen"
    
# Feeder population data uploaded
@app.callback(
    Output(component_id="feeder-population-data-file", component_property="children"),
    Input(component_id="feeder-population-data-upload", component_property="contents"),
    State(component_id="feeder-population-data-upload", component_property="filename")
)
def feeder_population_upload(contents, name):
    if contents is not None:
        return name
    else:
        return "No file chosen"
    
# Power factor adjusted
@app.callback(
    Output(component_id="power-factor-label", component_property="children"),
    Input(component_id="power-factor-slider", component_property="value")
)
def power_factor_update(value):
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
    State(component_id="max-c-rate-dropdown", component_property="value"),
    State(component_id="energy-cap-dropdown", component_property="value"),
    State(component_id="max-amp-hours-dropdown", component_property="value"),
    State(component_id="max-voltage-dropdown", component_property="value"),
    State(component_id="voltage-dropdown", component_property="value"),
    State(component_id="soh-input", component_property="value"),
    State(component_id="soc-input", component_property="value"),
    State(component_id="power-factor-slider", component_property="value"),
    State(component_id="battery-capacity-input", component_property="value"),
    State(component_id="feeder-population-data-upload", component_property="filename")
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
        max_c_rate,
        energy_cap,
        max_amp_hours,
        max_voltage,
        voltage,
        soh,
        soc,
        power_factor,
        battery_capacity,
        feeder_population_filename
    ):
    # TODO: Connect to backend here
    # either use preset_1, preset_2, or user_input depending on which is selected
    user_input = PRESET1 # TODO: create some config object from Preset 1
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
            user_input.battery["data"] = battery_filename
        else:
            user_input.ambient_data = temperature_filename
            user_input.load["data"] = load_filename
            user_input.solar["data"] = solar_filename
            user_input.solar["efficiency"] = solar_efficiency
            user_input.solar["rating"] = solar_capacity
            user_input.month = month
            # No provision for year
            user_input.num_days = num_days
            user_input.elec_prices["data"] = price_filename
            user_input.charging_station["dcfc_charging_stall_base_rating"] = dcfc_rating + "_kW"
            user_input.charging_station["l2_charging_stall_base_rating"] = l2_rating + "_kW"
            user_input.charging_station["num_dcfc_stalls_per_node"] = num_dcfc_stalls
            user_input.charging_station["num_l2_stalls_per_node"] = num_l2_stalls
            user_input.charging_station["commercial_building_trans"] = transformer_capactiy # is this the correct property?
            # TODO: fill in bettery dropdown values and format the values accordingly
            user_input.battery["max_c_rate"] = max_c_rate 
            user_input.battery["pack_energy_cap"] = energy_cap
            user_input.battery["pack_max_Ah"] = max_amp_hours
            user_input.battery["pack_max_voltage"] = max_voltage
            # Nothing for voltage
            # Nothing for SOC, SOH
            user_input.battery["power_factor"] = power_factor
            # Nothing for capacity
            if mpc_rhc_class == "setup-button selected":
                user_input.feeder_pop = True
                # Nothing for feeder pop data

    print(user_input)
    return {'grid-row': '2'}

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
