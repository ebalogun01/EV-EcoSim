from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd


## Dummy graph
def create_dummy_graph():
    data_canada = px.data.gapminder().query("country == 'Canada'")
    chart = create_bar_graph(data_canada, x='year', y='pop')
    return chart


## Create bar graph
def create_bar_graph(data, x=None, y=None):
    fig = px.bar(data,
                 x=x,
                 y=y,
                 color_discrete_sequence=px.colors.qualitative.Set1,)
    fig.update_xaxes(type='category')
    return dcc.Graph(figure=fig)


## Create line graph
def create_line_graph(data, x=None, y=None):
    fig = px.line(data,
                 x=x,
                 y=y,
                 color_discrete_sequence=px.colors.qualitative.Set1,)
    return dcc.Graph(figure=fig)
