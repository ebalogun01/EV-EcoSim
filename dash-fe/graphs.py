from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd


## Dummy graph
def create_dummy_graph():
    """
    Dummy placeholder graph creation

    :return: Dummy graph
    """
    data_canada = px.data.gapminder().query("country == 'Canada'")
    chart = create_bar_graph(data_canada, x='year', y='pop')
    return chart


## Create bar graph
def create_bar_graph(data, x=None, y=None, color=None):
    """
    Total bar graph creation

    :param data: source data
    :param x: x-axis
    :param y: y-axis
    :param color: color scheme of variables
    :return: Total bar graph
    """
    fig = px.bar(data,
                 x=x,
                 y=y,
                 color=color,
                 color_discrete_sequence=px.colors.qualitative.Plotly
                 )
    fig.update_xaxes(type='category')
    return dcc.Graph(figure=fig)

## Create bar graph
def create_total_bar_graph(data, x=None, y=None):
    """
    Bar graph creation

    :param data: source data
    :param x: x-axis
    :param y: y-axis
    :return: Bar graph
    """
    fig = px.bar(data,
                 x=x,
                 y=y,
                 color_discrete_sequence=px.colors.qualitative.Plotly
                 )
    fig.update_xaxes(type='category')
    return dcc.Graph(figure=fig)

## Create line graph
def create_line_graph(data, x=None, y=None):
    """
    Line graph creation

    :param data: source data
    :param x: x-axis
    :param y: y-axis
    :return: Line graph
    """
    fig = px.line(data,
                 x=x,
                 y=y,
                 #color_discrete_sequence=px.colors.qualitative.Set1
                 )
    return dcc.Graph(figure=fig)
