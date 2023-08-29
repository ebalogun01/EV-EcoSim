import plotly.express as px
import dash
from dash import dcc, html, ctx, Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from components import create_home_page, create_tutorial_page, create_output_page

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
                                            href="#",
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
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


# Preset 1 selected
@app.callback(
    Output(component_id="preset1-button", component_property="className"),
    Output(component_id="preset2-button", component_property="className"),
    Output(component_id="custom-settings-button", component_property="className"),
    Output(component_id="custom-settings-accordion", component_property="value"),
    Input(component_id="preset1-button", component_property="n_clicks"),
    Input(component_id="preset2-button", component_property="n_clicks"),
    Input(component_id="custom-settings-button", component_property="n_clicks"),
    Input(component_id="custom-settings-accordion", component_property="value"),
    prevent_initial_call=True
)
def select(preset1_n_clicks, preset2_n_clicks, custom_settings_n_clicks, custom_settings_value):
    triggered_id = ctx.triggered_id
    print(triggered_id)
    if triggered_id == "preset1-button":
        # Load preset 1
        return "setup-button selected", "setup-button", "setup-button", None
    elif triggered_id == "preset2-button":
        # Load preset 2
        return "setup-button", "setup-button selected", "setup-button", None
    elif triggered_id == "custom-settings-button":
        # Load custom settings
        return "setup-button", "setup-button", "setup-button selected", "customSettings"
    elif triggered_id == "custom-settings-accordion":
        # Load custom settings
        print(custom_settings_value)
        if custom_settings_value == "customSettings":
            return "setup-button", "setup-button", "setup-button selected", "customSettings"
        else:
            return
    else:
        return "setup-button", "setup-button", "setup-button"


if __name__ == '__main__':
    app.run_server(debug=True)
