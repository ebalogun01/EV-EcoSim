from dash import dcc, html, Input, Output
import plotly.express as px

def create_dummy_graph():
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x='year', y='pop')
    return dcc.Graph(figure=fig)