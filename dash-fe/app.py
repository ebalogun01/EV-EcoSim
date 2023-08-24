import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify

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
            style={'background-color': 'white', 'color': 'whitesmoke'},
            children=[
                dmc.Group(
                    position="left",
                    align="center",
                    children=[
                        dcc.Link(
                            dmc.ThemeIcon(
                                html.Img(src='assets/doerr.png', width="150"),
                                style={"paddingLeft": 74, "paddingTop": 28},
                                variant="filled",
                                color="white",
                            ),
                            href='https://sustainability.stanford.edu/',
                            target='_blank'
                        ),
                        dcc.Link(
                            href=app.get_relative_path("/"),
                            style={"paddingTop": 2, "paddingLeft": 150, "paddingBottom": 5, "paddingRight": 10,
                                   "textDecoration": "none"},
                            children=[
                                dmc.Group(align='center', spacing=0, position='center', children=[
                                    dmc.Text("EV-Ecosim", size="lg", color="gray",
                                             style={'font-family': 'Arial'}),
                                ]
                                          )
                            ]
                        ),
                        dmc.Group(
                            position="right",
                            align="center",
                            children=[
                                html.A(
                                    dmc.ThemeIcon(
                                        DashIconify(icon='mdi:github'),
                                        color='dark'
                                    ),
                                    href='https://github.com/ebalogun01/EV50_cosimulation',
                                    target='_blank'
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
                            options=[
                                {"label": "Input", "value": "INP"},
                                {"label": "Tutorial", "value": "TUT"},
                                {"label": "Output", "value": "OUT"}],
                            inline=True,
                        ),
                        html.Br(),
                    ]),
                ],
            ),
        ]),
    ]),
])


if __name__ == '__main__':
    app.run_server(debug=True)
