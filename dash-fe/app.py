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
            style={'background-color': '#fdd', 'color': 'whitesmoke'},
            children=[
                dmc.Group(
                    position="left",
                    align="center",
                    children=[
                        dcc.Link(
                            dmc.ThemeIcon(
                                html.Img(src='assets/doerr.png', width="150"),
                                style={"paddingLeft": 74, "paddingTop": 28},
                                variant="transparent",
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

                    html.Div(
                        id="tutorial-page",
                        style={
                            'display': 'block',
                        },
                        children=[
                            html.H2(
                                ["Tutorial"]
                            ),
                            html.P(
                                ["Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ipsum dolor sit amet consectetur. Velit euismod in pellentesque massa placerat duis ultricies lacus. Tempor id eu nisl nunc mi. Urna duis convallis convallis tellus id interdum velit. Pharetra diam sit amet nisl. Laoreet suspendisse interdum consectetur libero. Aliquet bibendum enim facilisis gravida neque convallis a cras. Eget mi proin sed libero enim sed. Non pulvinar neque laoreet suspendisse interdum consectetur. Sit amet mattis vulputate enim."]
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
                        ])
                ],
            ),
        ]),
    ]),
])

@app.callback(
    Output(component_id="tutorial-page", component_property="style"), 
    [Input(component_id="category-radio", component_property="value")])
def page_update(radio_value):
    print(radio_value)
    if radio_value == "TUT":
        return {'display': 'block'}
    else:
        return {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)
