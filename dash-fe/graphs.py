from dash import dcc, html, Input, Output
import plotly.express as px

## DUMMY GRAPH
def create_dummy_graph():
    data_canada = px.data.gapminder().query("country == 'Canada'")
    chart = create_bar_chart(data_canada, x='year', y='pop')
    return chart

## Create bar chart
def create_bar_chart(data, x, y):
    fig = px.bar(data, x=x, y=y)
    return dcc.Graph(figure=fig)

## TODO == PRICE SECTION ==
## TODO Levelized Cost of Energy (LCOE)

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