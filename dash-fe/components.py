import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import plotly.express as px
from graphs import *
import pandas as pd
from constants import TEXT


def make_input_section_label(grid_row, grid_column, icon, text, tooltip_text):
    return html.Div(
        className='setting-label tooltip',
        style={
            'grid-row': grid_row,
            'grid-column': grid_column,
        },
        children=[
            DashIconify(
                icon=icon,
                width=30,
                style={'margin': '0', 'margin-right': '10px'},
            ),
            html.H3(
                text,
                style={'margin': '0'},
            ),
            html.Span(
                className='tooltip-text',
                children=tooltip_text
            )
        ]
    )


def make_battery_dropdown(grid_row, grid_column, label, units, options, value, ):
    return html.Div(
        className='setup-dropdown-container',
        style={
            'grid-row': grid_row,
            'grid-column': grid_column
        },
        children=[
            html.Span(label),
            html.Div(
                style={
                    'display': 'flex',
                    'justify-content': 'space-between',
                },
                children=[
                    dcc.Dropdown(
                        className='setup-dropdown',
                        style={
                            'width': '100%'
                        },
                        options=options,
                        value=value,
                    ),
                    html.Span(
                        style={
                            'display': 'inline-block', 
                            'white-space': 'nowrap',
                            'padding-left': '12px'
                        }, 
                        children=[units]
                    )
                ]
            )
        ]
    )


def create_settings_container():
    settings_container = html.Div(
        id="settings-container",
        style={
            'display': 'grid',
            'grid-column-gap': '20px',
            'grid-row-gap': '20px',
            'grid-template-rows': 'repeat(2, auto)',
            'grid-template-columns': 'repeat(6, minmax(0, 1fr))'
        },
        children=[
            # Simulation mode
            make_input_section_label(grid_row='1', grid_column='2 / span 4',
                                     icon='heroicons:magnifying-glass-plus-20-solid', text='Simulation mode', tooltip_text='Tooltip'),
            html.Div(
                style={
                    'grid-row': '2',
                    'grid-column': '2 / span 4',
                    'display': 'flex',
                    'justify-content': 'space-between'
                },
                children=[
                    html.Button(
                        id="oneshot-button",
                        className='setup-button selected tooltip',
                        children=[
                            "One-shot optimization",
                            html.Span(
                                className='tooltip-text',
                                children='Tooltip'
                            )
                        ]
                    ),
                    html.Button(
                        id="mpc-rhc-button",
                        className="setup-button tooltip",
                        children=[
                            "MPC/RHC",
                            html.Span(
                                className='tooltip-text',
                                children='Tooltip'
                            )
                        ]
                    ),
                    html.Button(
                        id="battery-system-button",
                        className="setup-button tooltip",
                        children=[
                            "Battery system identification",
                            html.Span(
                                className='tooltip-text',
                                children='Tooltip'
                            )
                        ]
                    ),
                ]
            ),

            html.Div(
                id='simulation-container',
                style={
                    'display': 'grid',
                    'grid-row': '3',
                    'grid-column': '1 / span 6',
                    'grid-column-gap': '20px',
                    'grid-row-gap': '20px',
                    'grid-template-rows': 'repeat(20, auto)',
                    'grid-template-columns': 'repeat(6, minmax(0, 1fr))'
                },
                children=[
                    # Exogenous inputs
                    html.Div(
                        className='group-label',
                        style={
                            'grid-row': '1',
                            'grid-column': '1 / span 3'
                        },
                        children=[
                            html.H3("Exogenous inputs")
                        ]
                    ),

                    # Ambient temperature data
                    make_input_section_label(grid_row='2', grid_column='1 / span 3', icon='ph:thermometer-simple-bold',
                                            text='Ambient temperature data', tooltip_text='Tooltip'),
                    html.Div(
                        className='upload-container',
                        style={
                            'grid-row': '3',
                            'grid-column': '1  / span 3'
                        },
                        children=[
                            dcc.Upload(
                                id='temperature-data-upload',
                                children=[
                                    html.Button(
                                        children='Choose file'
                                    ),
                                    html.Span(
                                        id='temperature-data-file',
                                        children='No file chosen'
                                    )
                                ],
                            ),
                        ]
                    ),

                    # Solar
                    make_input_section_label(grid_row='2', grid_column='4 / span 3', icon='fa6-solid:solar-panel',
                                            text='Solar', tooltip_text='Tooltip'),
                    html.Div(
                        className='upload-container',
                        style={
                            'grid-row': '3',
                            'grid-column': '4 / span 3'
                        },
                        children=[
                            dcc.Upload(
                                id='solar-data-upload',
                                children=[
                                    html.Button('Choose file'),
                                    html.Span(
                                        id='solar-data-file',
                                        children='No file chosen'
                                    )
                                ],
                            ),
                        ]
                    ),
                    make_battery_dropdown(grid_row='4', grid_column='4 / span 3', label='Efficiency', units='units',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1'
                                        ),
                    make_battery_dropdown(grid_row='5', grid_column='4 / span 3', label='Capacity', units='units',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1'
                                        ),

                    # Load
                    make_input_section_label(grid_row='4', grid_column='1 / span 3', icon='icon-park-solid:screenshot-one',
                                            text='Load', tooltip_text='Tooltip'),
                    html.Div(
                        className='upload-container',
                        style={
                            'grid-row': '5',
                            'grid-column': '1 / span 3'
                        },
                        children=[
                            dcc.Upload(
                                id='load-data-upload',
                                children=[
                                    html.Button('Choose file'),
                                    html.Span(
                                        id='load-data-file', 
                                        children='No file chosen'
                                    )
                                ],
                            ),
                        ]
                    ),

                    # System configurations
                    html.Div(
                        className='group-label',
                        style={
                            'grid-row': '6',
                            'grid-column': '1 / span 3'
                        },
                        children=[
                            html.H3("System configurations")
                        ]
                    ),
                    
                    # Timescale
                    make_input_section_label(grid_row='7', grid_column='1 / span 3',
                                            icon='heroicons:magnifying-glass-plus-20-solid', text='Timescale', tooltip_text='Tooltip'),
                    html.Div(
                        style={
                            'grid-row': '8',
                            'grid-column': '1 / span 3',
                            'display': 'flex',
                            'justify-content': 'space-between',
                        },
                        children=[
                            dmc.DatePicker(
                                id="start-date-picker",
                                className="setup-date-picker",
                                inputFormat="MM/DD/YYYY",
                            ),
                            html.Span(
                                style={
                                    'display': 'inline-grid',
                                    'align-items': 'center',
                                    'text-align': 'center',
                                    'margin': '0 8px'
                                },
                                children=["-"]
                            ),
                            dmc.DatePicker(
                                id="end-date-picker",
                                className="setup-date-picker",
                                inputFormat="MM/DD/YYYY",
                            ),
                        ]
                    ),

                    # Electricity price
                    make_input_section_label(grid_row='7', grid_column='4 / span 3', icon='fa6-solid:car-battery',
                                            text='Electricity price', tooltip_text='Tooltip'),
                    html.Div(
                        className='upload-container',
                        style={
                            'grid-row': '8',
                            'grid-column': '4 / span 3'
                        },
                        children=[
                            dcc.Upload(
                                id='price-data-upload',
                                children=[
                                    html.Button('Choose file'),
                                    html.Span(
                                        id='price-data-file',
                                        children='No file chosen'
                                    )
                                ],
                            ),
                        ]
                    ),

                    # Charging station
                    make_input_section_label(grid_row='9', grid_column='1 / span 3', icon='carbon:charging-station-filled',
                                            text='Charging station', tooltip_text='Tooltip'),
                    make_battery_dropdown(grid_row='10', grid_column='1 / span 3', label='DCFC stall rating', units='kW',
                                        options=[
                                            {'label': '75', 'value': '75'},
                                        ],
                                        value='75'
                                        ),
                    make_battery_dropdown(grid_row='11', grid_column='1 / span 3', label='L2 charging stall rating', units='kW',
                                        options=[
                                            {'label': '11.5', 'value': '11.5'},
                                        ],
                                        value='11.5'
                                        ),
                    make_battery_dropdown(grid_row='12', grid_column='1 / span 3', label='Number of DCFC stalls per node', units='',
                                        options=[
                                            {'label': '5', 'value': '5'},
                                        ],
                                        value='5'
                                        ),
                    make_battery_dropdown(grid_row='13', grid_column='1 / span 3', label='Number of L2 stalls per node', units='',
                                        options=[
                                            {'label': '0', 'value': '0'},
                                        ],
                                        value='0'
                                        ),
                    make_battery_dropdown(grid_row='14', grid_column='1 / span 3', label='Transformer capacity', units='kVA',
                                        options=[
                                            {'label': '75', 'value': '75'},
                                        ],
                                        value='75'
                                        ),

                    # Battery
                    make_input_section_label(grid_row='9', grid_column='4 / span 3', icon='clarity:battery-solid',
                                            text='Battery', tooltip_text='Tooltip'),
                    make_battery_dropdown(grid_row='10', grid_column='4 / span 3', label='Maximum C-rate', units='',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1'
                                        ),
                    make_battery_dropdown(grid_row='11', grid_column='4 / span 3', label='Energy capacity', units='units',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1'
                                        ),
                    make_battery_dropdown(grid_row='12', grid_column='4 / span 3', label='Maximum amp hours', units='Ah',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1'
                                        ),
                    make_battery_dropdown(grid_row='13', grid_column='4 / span 3', label='Maximum voltage', units='V',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1'
                                        ),
                    make_battery_dropdown(grid_row='14', grid_column='4 / span 3', label='Voltage', units='V',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1'
                                        ),
                    make_battery_dropdown(grid_row='15', grid_column='4 / span 3', label='State of health', units='',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1'
                                        ),
                    make_battery_dropdown(grid_row='16', grid_column='4 / span 3', label='State of charge', units='',
                                        options=[
                                            {'label': 'Value 1', 'value': '1'},
                                            {'label': 'Value 2', 'value': '2'},
                                            {'label': 'Value 3', 'value': '3'}
                                        ],
                                        value='1',
                                        ),

                    # Power factor
                    make_input_section_label(grid_row='15', grid_column='1 / span 3', icon='bi:cloud-lightning-fill',
                                            text='Power factor (optional)', tooltip_text='Tooltip'),
                    html.Div(
                        style={
                            'grid-row': '16',
                            'grid-column': '1 / span 3',
                            'display': 'flex',
                            'justify-content': 'space-between',
                        },
                        children=[
                            html.Div(
                                dcc.Slider(
                                    id="power-factor-slider",
                                    min=0,
                                    max=100,
                                    step=10,
                                    value=None,
                                    included=True,
                                ),
                                className='slider-container',
                                style={
                                    'width': '100%',
                                    'display': 'inline-grid',
                                    'align-items': 'center',
                                }
                            ),
                            html.Span(
                                id="power-factor-label",
                                className="slider-label",
                                children=["XX"]
                            )
                        ]
                    ),

                    # Capacity
                    make_input_section_label(grid_row='17', grid_column='1 / span 3', icon='material-symbols:screenshot-frame',
                                            text='Capacity', tooltip_text='Tooltip'),
                    html.Div(
                        style={
                            'grid-row': '18',
                            'grid-column': '1 / span 3',
                            'display': 'flex',
                            'justify-content': 'space-between',
                        },
                        children=[
                            html.Div(
                                dcc.Slider(
                                    id='capacity-slider',
                                    min=0,
                                    max=100,
                                    step=10,
                                    value=50,
                                    included=True,
                                ),
                                className='slider-container',
                                style={
                                    'width': '100%',
                                    'display': 'inline-grid',
                                    'align-items': 'center'
                                }
                            ),
                            html.Span(
                                id="capacity-label",
                                className="slider-label",
                                children=["XX"]
                            )
                        ]
                    ),

                    # Feeder population
                    html.Div(
                        id='feeder-population-container',
                        style={
                            'display': 'none',
                            'grid-row-gap': '20px',
                            'grid-row': '17 / span 2',
                            'grid-column': '4 / span 3'
                        },
                        children=[
                            make_input_section_label(grid_row='1', grid_column='1', icon='fa6-regular:window-restore',
                                                    text='Feeder population', tooltip_text='Tooltip'),
                            html.Div(
                                className='upload-container',
                                style={
                                    'grid-row': '2',
                                    'grid-column': '1'
                                },
                                children=[
                                    dcc.Upload(
                                        id="feeder-population-data-upload",
                                        children=[
                                            html.Button('Choose file'),
                                            html.Span(
                                                id='feeder-population-data-file',
                                                children='No file chosen'
                                            )
                                        ]
                                    ),
                                ]
                            ),
                        ]

                    ),

                ]
            ),

            # File upload
            html.Div(
                id='battery-system-container',
                style={
                    'display': 'none',
                },
                children=[
                    html.Div(
                        className='upload-container',
                        children=[
                            dcc.Upload(
                                id='battery-data-upload',
                                children=[
                                    html.Button('Choose file'),
                                    html.Span(
                                        id='battery-data-file',
                                        children='No file chosen'
                                    )
                                ]
                            ),
                        ]
                    ),
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
                    'grid-template-rows': 'auto',
                    'grid-template-columns': 'repeat(2, auto)',
                    'grid-column-gap': '20px'
                },
                children=[
                    html.Div(
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

            # Simulation setup tile
            html.Div(
                className="content-tile",
                style={
                    'grid-template-rows': 'repeat(2, auto)',
                    'grid-template-columns': 'repeat(4, minmax(0, 1fr))'
                },
                children=[
                    html.H3(
                        className="section-title",
                        children=["Simulation setup"]
                    ),
                    html.Button(
                        id="preset1-button",
                        className='setup-button selected tooltip',
                        style={'grid-row': '2'},
                        children=[
                            "Preset 1",
                            html.Span(
                                className='tooltip-text',
                                children='Tooltip'
                            )
                        ]
                    ),
                    html.Button(
                        id="preset2-button",
                        className='setup-button tooltip',
                        style={'grid-row': '2'},
                        children=[
                            "Preset 2",
                            html.Span(
                                className='tooltip-text',
                                children='Tooltip'
                            )
                        ]
                    ),
                    html.Button(
                        id="custom-settings-button",
                        className='setup-button tooltip',
                        style={'grid-row': '2'},
                        children=[
                            "Custom configuration",
                            html.Span(
                                className='tooltip-text',
                                children='Tooltip'
                            )
                        ]
                    ),
                    html.Button(
                        id='run-simulation-button',
                        className='action tooltip',
                        style={'grid-row': '2'},
                        children=[
                            "Run simulation",
                            html.Span(
                                className='tooltip-text',
                                children='Tooltip'
                            )
                        ]
                    ),
                ]
            ),

            # Custom configuration tile
            html.Div(
                className="content-tile",
                style={
                    'grid-template-rows': 'auto',
                    'grid-template-columns': 'auto'
                },
                children=[
                    dmc.Accordion(
                        id="custom-settings-accordion",
                        chevronPosition="left",
                        value=None,
                        children=[
                            dmc.AccordionItem(
                                [
                                    dmc.AccordionControl(
                                        html.H3(
                                            className="section-title",
                                            children=["Custom configuration"]
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
    ## TODO == PRICE SECTION ==
    ## TODO custom data

    ## LCOE
    #lcoe_data = pd.read_csv('data/dummy/costs-June-oneshot-collated-results/Total_June_costs_per_day.csv')
    lcoe_data = pd.read_csv('dash-fe/data/dummy/costs-June-oneshot-collated-results/Total_June_costs_per_day.csv')
    lcoe_data = lcoe_data.rename(columns={'Unnamed: 0': 'c'})
    lcoe_data = lcoe_data.filter(['c', '50.0'], axis="columns")

    ## Battery aging costs
    #bat_age_data = pd.read_csv('data/dummy/costs-June-oneshot-collated-results/June_battery_aging_costs_per_day.csv')
    bat_age_data = pd.read_csv('dash-fe/data/dummy/costs-June-oneshot-collated-results/June_battery_aging_costs_per_day.csv')
    bat_age_data = bat_age_data.rename(columns={'Unnamed: 0': 'c'})
    bat_age_data = bat_age_data.filter(['c', '50.0'], axis="columns")

    ## Battery costs
    #bat_cost_data = pd.read_csv('data/dummy/costs-June-oneshot-collated-results/June_battery_costs_per_day.csv')
    bat_cost_data = pd.read_csv('dash-fe/data/dummy/costs-June-oneshot-collated-results/June_battery_costs_per_day.csv')
    bat_cost_data = bat_cost_data.rename(columns={'Unnamed: 0': 'c'})
    bat_cost_data = bat_cost_data.filter(['c', '50.0'], axis="columns")

    ## Transformer aging costs
    #tra_age_data = pd.read_csv('data/dummy/costs-June-oneshot-collated-results/June_trans_aging_per_day.csv')
    tra_age_data = pd.read_csv('dash-fe/data/dummy/costs-June-oneshot-collated-results/June_trans_aging_per_day.csv')
    tra_age_data = tra_age_data.rename(columns={'Unnamed: 0': 'c'})
    tra_age_data = tra_age_data.filter(['c', '50.0'], axis="columns")

    ## Electricity costs
    #elec_data = pd.read_csv('data/dummy/costs-June-oneshot-collated-results/June_elec_costs_per_day.csv')
    elec_data = pd.read_csv('dash-fe/data/dummy/costs-June-oneshot-collated-results/June_elec_costs_per_day.csv')
    elec_data = elec_data.rename(columns={'Unnamed: 0': 'c'})
    elec_data = elec_data.filter(['c', '50.0'], axis="columns")

    ## LCOE
    bat_age_data['type'] = 'Battery aging'
    bat_cost_data['type'] = 'Battery costs'
    tra_age_data['type'] = 'Transformer aging'
    elec_data['type'] = 'Electricity costs'
    lcoe_data = pd.concat(
        [bat_age_data, bat_cost_data, tra_age_data, elec_data],
        axis=0,
        join="outer",
        ignore_index=False,
        keys=None,
        levels=None,
        names=None,
        verify_integrity=False,
        copy=True,
    )

    price_section = dmc.Card(
        style={"padding": "5px"},
        children=[
            create_badge_title("Price", 'mdi:money'),
            create_graph_card(
                title="Levelized Cost of Energy (LCOE)",
                data=lcoe_data,
                bar_color="type",
                download_link="#",
            ),
            dmc.Group(
                position="center",
                grow=True,
                children=[
                    create_graph_card(
                        title="Battery aging costs",
                        data=bat_age_data,
                        download_link="#"
                    ),
                    create_graph_card(
                        title="Battery costs",
                        data=bat_cost_data,
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
                        data=tra_age_data,
                        download_link="#"
                    ),
                    create_graph_card(
                        title="Electricity costs",
                        data=elec_data,
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
    ## TODO == CHARGING STATION SECTION ==

    ## Charging station data setup
    #charging_station_data = pd.read_csv('data/dummy/battery-transformers-June15-oneshot/charging_station_sim_0_dcfc_load_0.csv')
    charging_station_data = pd.read_csv('dash-fe/data/dummy/battery-transformers-June15-oneshot/charging_station_sim_0_dcfc_load_0.csv')
    ## Net grid load
    ngl_data = charging_station_data.filter(['station_net_grid_load_kW'], axis="columns")
    ## Total load
    tl_data = charging_station_data.filter(['station_total_load_kW'], axis="columns")

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
                        data=tl_data,
                        x=None,
                        y='station_total_load_kW',
                        graph_type='line',
                        download_link="#"
                    ),
                    create_graph_card(
                        title="Net grid load",
                        data=ngl_data,
                        x=None,
                        y='station_net_grid_load_kW',
                        graph_type='line',
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
    ## TODO == BATTERY SECTION ==

    ## Battery data setup
    #battery_data = pd.read_csv('data/dummy/battery-transformers-June15-oneshot/battery_sim_0_dcfc_load_0.csv')
    battery_data = pd.read_csv('dash-fe/data/dummy/battery-transformers-June15-oneshot/battery_sim_0_dcfc_load_0.csv')

    ## TODO State of charge (SOC)
    soc_data = battery_data.filter(['SOC'], axis="columns")

    ## TODO Current
    current_data = battery_data.filter(['currents_pack'], axis="columns")

    ## TODO Voltage
    voltage_data = battery_data.filter(['Voltage_pack'], axis="columns")

    ## TODO Power
    power_data = battery_data.filter(['power_kW'], axis="columns")

    ## TODO State of health (SOH)
    soh_data = battery_data.filter(['SOH'], axis="columns")

    battery_section = dmc.Card(
        style={"padding": "5px"},
        children=[
            create_badge_title("Battery", 'clarity:battery-solid'),
            create_graph_card(
                title="State of charge",
                data=soc_data,
                x=None,
                y='SOC',
                graph_type='line',
                download_link="#"
            ),
            dmc.Group(
                position="center",
                grow=True,
                children=[
                    create_graph_card(
                        title="Current",
                        data=current_data,
                        x=None,
                        y='currents_pack',
                        graph_type='line',
                        download_link="#"
                    ),
                    create_graph_card(
                        title="Voltage",
                        data=voltage_data,
                        x=None,
                        y='Voltage_pack',
                        graph_type='line',
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
                        data=power_data,
                        x=None,
                        y='power_kW',
                        graph_type='line',
                        download_link="#"
                    ),
                    create_graph_card(
                        title="State of health",
                        data=soh_data,
                        x=None,
                        y='SOH',
                        graph_type='line',
                        download_link="#"
                    )
                ]
            )
        ],
        withBorder=True,
        shadow="md",
    )
    return battery_section


def create_graph_card(title="Undefined title", description="Undefined description", data=None, graph_type='bar',
                      bar_color=None,
                      download_link=None, x='c', y='50.0'):
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
                                     graph_type=graph_type,
                                     x=x,
                                     y=y,
                                     color=bar_color
                                     )
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


def create_graph_element(data=None, graph_type='bar', color=None, x='c', y='50.0'):
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
                              children=create_bar_graph(data, x=x, y=y, color=color)
                              ),
            mb=10,
        )
    elif graph_type == 'line':
        return dmc.Skeleton(
            visible=False,
            children=html.Div(className="graph-container",
                              children=create_line_graph(data, x=x, y=y)
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