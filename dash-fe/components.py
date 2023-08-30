import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
import plotly.express as px
from constants import TEXT
from dash_iconify import DashIconify
from graphs import *
import pandas as pd

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
                                    dmc.AccordionPanel("ASDF"),
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
                    # TODO add scenarios and scenario comparison
                    create_price_section(),
                    create_charging_section(),
                    create_battery_section()
                ]
            ),
        ]
    )
    return output_page


def create_price_section():
    # TODO custom data
    lcoe_data = pd.DataFrame(pd.read_csv('data/dummy/costs-June-oneshot-collated-results/Total_June_costs_per_day.csv'))
    lcoe_data = pd.DataFrame(lcoe_data)

    price_section = dmc.Card(
        style={"padding": "5px"},
        children=[
            create_badge_title("Price", 'mdi:money'),
            create_graph_card(
                title="Levelized Cost of Energy (LCOE)",
                data=lcoe_data,
                download_link="#",
            ),
            dmc.Group(
                position="center",
                grow=True,
                children=[
                    create_graph_card(
                        title="Battery aging costs",
                        download_link="#"
                    ),
                    create_graph_card(
                        title="Battery costs",
                        download_link="#"
                    )
                ]
            ),
            dmc.Group(
                position="center",
                grow=True,
                children=[
                    create_graph_card(
                        title="Transformer aging costs",
                        download_link="#"
                    ),
                    create_graph_card(
                        title="Electricity costs",
                        download_link="#"
                    )
                ]
            )
        ],
        withBorder=True,
        shadow="md",
    )
    return price_section


def create_charging_section():
    charging_section = dmc.Card(
        style={"padding": "5px"},
        children=[
            create_badge_title("Charging station", 'fa6-solid:charging-station'),
            dmc.Group(
                position="center",
                grow=True,
                children=[
                    create_graph_card(
                        title="Total load",
                        download_link="#"
                    ),
                    create_graph_card(
                        title="Net grid load",
                        download_link="#"
                    )
                ]
            )
        ],
        withBorder=True,
        shadow="md",
    )
    return charging_section


def create_battery_section():
    battery_section = dmc.Card(
        style={"padding": "5px"},
        children=[
            create_badge_title("Battery", 'clarity:battery-solid'),
            create_graph_card(
                title="State of charge",
                download_link="#"
            ),
            dmc.Group(
                position="center",
                grow=True,
                children=[
                    create_graph_card(
                        title="Current",
                        download_link="#"
                    ),
                    create_graph_card(
                        title="Voltage",
                        download_link="#"
                    )
                ]
            ),
            dmc.Group(
                position="center",
                grow=True,
                children=[
                    create_graph_card(
                        title="Power",
                        download_link="#"
                    ),
                    create_graph_card(
                        title="State of health",
                        download_link="#"
                    )
                ]
            )
        ],
        withBorder=True,
        shadow="md",
    )
    return battery_section


def create_graph_card(title="Undefined title", description="Undefined description", data=None, graph_type='bar',download_link=None):
    card = dmc.Card(
        style={"margin": "2px"},
        children=[
            dmc.Group(
                style={"margin": "5px"},
                children=[
                    dmc.Text(title,
                             weight=700,
                             style={"margin-top": "-20px"}, ),
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
                                href=download_link,
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
                create_graph_element(data=data,
                                     graph_type=graph_type)
            ),
            dmc.Text(
                description,
                size="sm",
                color="dimmed",
            ),
        ],
        withBorder=False,
        shadow="md",
        radius="md",
    )
    return card

def create_graph_element(data = None, graph_type = 'bar'):
    if data is None:
        return dmc.Skeleton(
            visible=True,
            children=html.Div(className="graph-container",
                              children=create_dummy_graph()
                              ),
            mb=10,
        )
    elif graph_type == 'bar':
        return dmc.Skeleton(
            visible=False,
            children=html.Div(className="graph-container",
                              children=create_bar_graph(data, x='50.0', y='100.0')
                              ),
            mb=10,
        )
    else:
        return None

def create_badge_title(title, icon):
    return dmc.Group(
        style={"margin": "5px"},
        children=[
            dmc.Badge(children=[title],
                      size="xl",
                      variant="filled",
                      color="gray",
                      radius=4,
                      leftSection=dmc.ThemeIcon(
                          children=DashIconify(icon=icon, width=25),
                          color='gray',
                          size='lg'),
                      sx={"paddingLeft": 5},
                      style={"margin-top": "5px"},
                      ),
        ]
    )
