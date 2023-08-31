from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd


## Dummy graph
def create_dummy_graph():
    data_canada = px.data.gapminder().query("country == 'Canada'")
    chart = create_bar_graph(data_canada, x='year', y='pop')
    return chart


## Create bar graph
def create_bar_graph(data, x, y):
    fig = px.bar(data,
                 x=x,
                 y=y,
                 color_discrete_sequence=px.colors.qualitative.Set1,)
    return dcc.Graph(figure=fig)


## Create line graph
def create_line_graph(data, x, y):
    fig = px.line(data,
                 x=x,
                 y=y,
                 color_discrete_sequence=px.colors.qualitative.Set1,)
    return dcc.Graph(figure=fig)


## TODO == PRICE SECTION ==
## Levelized Cost of Energy (LCOE)
def create_lcoe_chart(data=None):
    # Example data if run without data
    if data == None:
        # TODO add custom data
        src = pd.read_csv('data/dummy/costs-June-oneshot-collated-results/Total_June_costs_per_day.csv')
        dummy_data = src[:][1]
        return create_bar_chart(dummy_data)
    # TODO custom data
    else:
        return create_bar_chart(data)

## TODO Battery aging costs

## TODO Battery costs

## TODO Transformer aging costs

## TODO Electricity costs


## TODO == CHARGING STATION SECTION ==
## TODO Net grid load

## TODO Total load

## TODO == BATTERY SECTION ==
## TODO State of charge (SOC)

## TODO Current

## TODO Voltage

## TODO Power

## TODO State of health (SOH)
