import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.express as px
from constants import TEXT
from dash_iconify import DashIconify

def create_settings_container():
  settings_container = html.Div(
    style={
      'display': 'grid',
      'grid-column-gap': '20px',
      'grid-row-gap': '20px',
      'grid-template': 'auto auto auto auto auto auto auto auto auto auto auto auto / auto auto'
    },
    children=[
      html.Div(
        className="setting-label",
        style={
          'grid-row-start': '1',
          'grid-column-start': '1',
        },
        children=[
          DashIconify(
            icon="heroicons:magnifying-glass-plus-20-solid", 
            width=30,
            style={
              'margin': '0 auto'
            }
          ),
          html.H3(
            style={
              'margin': '0 auto'
            },
            children=["Simulation mode"]
          )
        ]
      ),

      html.Div(
        className="setting-label",
        style={
          'grid-row-start': '1',
          'grid-column-start': '2',
        },
        children=[
          DashIconify(
            icon="heroicons:magnifying-glass-plus-20-solid", 
            width=30,
            style={
              'margin': '0 auto'
            }
          ),
          html.H3(
            style={
              'margin': '0 auto'
            },
            children=["Timescale"]
          )
        ]
      ),

      html.Div(
        style={
          'grid-row-start': '2',
          'grid-column-start': '1',
          'display': 'flex',
          'justify-content': 'space-between'
        },
        children=[
          html.Button(
            style={'width': '48%'},
            id="offline-button",
            className='setup-button selected',
            children=["Offline"]
          ),
          html.Button(
            style={'width': '48%'},
            id="mpc-button",
            className="setup-button",
            children=["MPC"]
          )
        ]
      )
    ]
  )

  return settings_container

def create_settings_container():
  settings_container = html.Div(
    style={
      'display': 'grid',
      'grid-column-gap': '20px',
      'grid-row-gap': '20px',
      'grid-template': 'auto auto auto auto auto auto auto auto auto auto auto auto / auto auto'
    },
    children=[
      html.Div(
        className="setting-label",
        style={
          'grid-row-start': '1',
          'grid-column-start': '1',
        },
        children=[
          DashIconify(
            icon="heroicons:magnifying-glass-plus-20-solid", 
            width=30,
            style={
              'margin': '0 auto'
            }
          ),
          html.H3(
            style={
              'margin': '0 auto'
            },
            children=["Simulation mode"]
          )
        ]
      ),

      html.Div(
        className="setting-label",
        style={
          'grid-row-start': '1',
          'grid-column-start': '2',
        },
        children=[
          DashIconify(
            icon="heroicons:magnifying-glass-plus-20-solid", 
            width=30,
            style={
              'margin': '0 auto'
            }
          ),
          html.H3(
            style={
              'margin': '0 auto'
            },
            children=["Timescale"]
          )
        ]
      ),

      html.Div(
        style={
          'grid-row-start': '2',
          'grid-column-start': '1',
          'display': 'flex',
          'justify-content': 'space-between'
        },
        children=[
          html.Button(
            style={'width': '48%'},
            id="offline-button",
            className='setup-button selected',
            children=["Offline"]
          ),
          html.Button(
            style={'width': '48%'},
            id="mpc-button",
            className="setup-button",
            children=["MPC"]
          )
        ]
      )
    ]
  )

  return settings_container

def create_home_page():
    home_page = html.Div(
        id='home-page',
        style={'display': 'block'},
        children=[
            # Page title
            html.H2(
                className="page-title",
                children=["Welcome to EV-Ecosim"]
            ),

            # Tutorial/how to use tile
            html.Div(
                className="content-tile",
                style={
                    'grid-template': 'auto / auto auto'
                },
                children=[
                    html.Div(
                        id="introduction-section",
                        style={'padding': '12px'},
                        children=[
                            html.H3(
                                className="section-title",
                                children=["Introduction"]
                            ),
                            html.P(
                                children=[TEXT['intro']]
                            ),
                            html.Button(
                                className="action",
                                children=["See tutorial"]
                            )
                        ]
                    ),
                    html.Div(
                        id="how-to-use-section",
                        style={'padding': '12px'},
                        children=[
                            html.H3(
                                className="section-title",
                                children=["How to use"]
                            ),
                            html.P(
                                children=[TEXT['howToUse']]
                            )
                        ]
                    ),
                ]
            ),

            # Input buttons tile
            html.Div(
                className="content-tile",
                style={
                    'grid-template': 'auto auto / auto auto auto auto'
                },
                children=[
                    html.H3(
                        className="section-title",
                        children=["Simulation setup"]
                    ),
                    html.Button(
                        id="preset1-button",
                        className='setup-button selected',
                        style={'grid-row-start': '2'},
                        children=["Preset 1"]
                    ),
                    html.Button(
                        id="preset2-button",
                        className='setup-button',
                        style={'grid-row-start': '2'},
                        children=["Preset 2"]
                    ),
                    html.Button(
                        id="custom-settings-button",
                        className='setup-button',
                        style={'grid-row-start': '2'},
                        children=["Custom settings"]
                    ),
                    html.Button(
                        className='action',
                        style={'grid-row-start': '2'},
                        children=["Run simulation"]
                    ),
                ]
            ),

      # Custom configuration tile
      html.Div(
        className="content-tile",
        style={
          'grid-template': 'auto / auto'
        },
        children=[
          dmc.Accordion(
            id="custom-settings-accordion",
            chevronPosition="left",
            value="none",
            children=[
              dmc.AccordionItem(
                [
                  dmc.AccordionControl(
                    html.H3(
                      className="section-title",
                      children=["Custom setup settings"]
                    )
                  ),
                  dmc.AccordionPanel([create_settings_container()]),
                ],
                value="customSettings"
              )
            ]
          ),
        ]
      ),

            # Credits tile
            html.Div(
                className="content-tile",
                style={
                    'grid-template': 'auto auto auto auto / auto'
                },
                children=[
                    html.H3(
                        className="section-title",
                        children=["Credits, How to cite"]
                    ),
                    html.P(
                        children=[TEXT['credits1']]
                    ),
                    html.P(
                        children=[TEXT['credits2']]
                    ),
                    html.P(
                        children=[TEXT['credits3']]
                    )
                ]
            ),
        ]
    )

    return home_page


def create_tutorial_page():
    tutorial_page = html.Div(
        id="tutorial-page",
        style={
            'display': 'none',
        },
        children=[
            # Page title
            html.Div(
                style={'position': 'relative'},
                children=[
                    html.H2(
                        style={'display': 'inline'},
                        className="page-title",
                        children=["EV-Ecosim tutorial"]
                    ),
                    html.Button(
                        style={
                            'width': '200px',
                            'position': 'absolute',
                            'right': '0'
                        },
                        className="action",
                        children=["Back to home"]
                    ),
                ]
            ),

            # Page content
            html.Div(
                className="content-tile",
                style={
                    'grid-template': 'auto / auto'
                },
                children=[
                    html.H3(
                        className="section-title",
                        children=["Tutorial"]
                    ),
                    html.P(
                        TEXT['intro']
                    ),
                    html.Iframe(
                        id="tutorial-video",
                        width="627",
                        height="352.5",
                        style={
                            'display': 'block',
                            'margin': 'auto'
                        },
                        src="https://www.youtube.com/embed/MH_3H8uHxF0",
                        title="Learn about Plotly &amp; Dash",
                    ),
                    html.H3(
                        className="section-title",
                        children=["How to use"]
                    ),
                    html.P(
                        TEXT['howToUse']
                    ),
                    html.Button(
                        style={'width': '200px'},
                        className="action",
                        children=["Back to home"]
                    ),
                ]
            ),
        ]
    )

    return tutorial_page


def create_output_page():
    output_page = html.Div(
        id="output-page",
        style={
            'display': 'none',
        },
        children=[
            # Page title
            html.Div(
                style={'position': 'relative'},
                children=[
                    html.Div(
                        [
                            html.H2(
                                style={'display': 'inline'},
                                className="page-title",
                                children=["EV-Ecosim output"]
                            ),
                            html.P(
                                html.H5(
                                    style={'display': 'inline'},
                                    className="page-title",
                                    children=["For input: xxxxxx"]
                                ),
                            ),
                        ],
                    ),

                ]
            ),

            # Page content
            html.Div(
                className="content-tile",
                style={
                    'grid-template': 'auto / auto'
                },
                children=[
                    dmc.Card(
                        children=[
                            dmc.Group(
                                [
                                    dmc.Text("Levelized Cost of Energy (LCOE)", weight=700),
                                    dmc.Group(
                                        position="right",
                                        align="right",
                                        children=[
                                            html.A(
                                                dmc.Badge(
                                                    "Download data (CSV)",
                                                    leftSection=dmc.ThemeIcon(
                                                        DashIconify(icon='mdi:download'),
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
                                position="apart",
                                mt="md",
                                mb="xs",
                            ),
                            dmc.CardSection(
                                dmc.Skeleton(
                                    visible=True,
                                    children=html.Div(id="skeleton-graph-container",
                                                      children=create_dummy_graph()
                                                      ),
                                    mb=10,
                                ),
                            ),
                            dmc.Text(
                                "Description",
                                size="sm",
                                color="dimmed",
                            ),
                        ],
                        withBorder=False,
                        shadow="sm",
                        radius="md",
                    ),
                ]
            ),
        ]
    )
    return output_page

def create_dummy_graph():
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x='year', y='pop')
    return dcc.Graph(figure=fig)
