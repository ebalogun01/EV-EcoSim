import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
from constants import TEXT

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
          'grid-template':'auto auto auto auto / auto'
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