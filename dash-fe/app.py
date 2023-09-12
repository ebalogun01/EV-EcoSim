import plotly.express as px
import dash
from dash import dcc, html, ctx, Input, Output, State
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from components import create_home_page, create_tutorial_page, create_output_page
from constants import PRESET1, PRESET2, CUSTOM_DEFAULT

# Create Dash app
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

# Create input dictionary

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
        user_input = PRESET1
        return "setup-button selected tooltip", "setup-button tooltip", "setup-button tooltip", None
    elif triggered_id == "preset2-button":
        # Load preset 2
        user_input = PRESET2
        return "setup-button tooltip", "setup-button selected tooltip", "setup-button tooltip", None
    elif triggered_id == "custom-settings-button":
        user_input = CUSTOM_DEFAULT
        return "setup-button tooltip", "setup-button tooltip", "setup-button selected tooltip", "customSettings"
    elif triggered_id == "custom-settings-accordion":
        # Load custom settings
        print(custom_settings_value)
        if custom_settings_value == "customSettings":
            user_input = CUSTOM_DEFAULT
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
    # Include state properties needed to run simulation
)
def run_simulation(run_button_n_clicks):
    # TODO: Connect to backend here
    return {'grid-row': '2'}

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
