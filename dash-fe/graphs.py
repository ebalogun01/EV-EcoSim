from dash import dcc
import plotly.express as px
import plotly.graph_objs as go


# Dummy graph
def create_dummy_graph():
    """
    Dummy placeholder graph creation

    :return: Dummy graph
    """
    data_canada = px.data.gapminder().query("country == 'Canada'")
    chart = create_bar_graph(data_canada, x='year', y='pop')
    return chart


# Create bar graph
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


# Create bar graph
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


# Create line graph
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
                  # color_discrete_sequence=px.colors.qualitative.Set1
                  )
    return dcc.Graph(figure=fig)


# Create heatmap
def create_heatmap(data, x=None, y=None):
    """
    Line graph creation

    :param data: source data
    :param x: x-axis
    :param y: y-axis
    :return: Heatmap
    """
    fig = px.imshow(data,
                    x=x,
                    y=y,
                    # aspect="auto", TODO optimize aspect ratio
                    color_continuous_scale='matter'
                    )
    fig.update_xaxes(type='category')
    fig.update_yaxes(type='category')
    return dcc.Graph(figure=fig)


# Create 4D graph
def create_4D_graph(data, x=None, y=None, z=None, q=None):
    """
    4D graph creation

    :param data: source data
    :param x: x-axis
    :param y: y-axis
    :param y: z-axis
    :param q: q-axis, shown as color
    :return: 4D graph
    """

    # Src: https://github.com/ostwalprasad/PythonMultiDimensionalPlots/blob/master/src/4D.py
    fig = go.Scatter3d(x=x,
                       y=y,
                       z=z,
                       marker=dict(color=q,
                                   opacity=1,
                                   reversescale=True,
                                   colorscale='Blues',
                                   size=5),
                       line=dict(width=0.02),
                       mode='markers')

    return dcc.Graph(figure=fig)
