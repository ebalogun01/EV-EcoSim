import dash
from dash import dcc, html, Input, Output
from constants import TEXT

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
