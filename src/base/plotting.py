import random

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets

from base.results import Results


def to_jupyter_widget(fig: go.Figure) -> go.FigureWidget:
    return go.FigureWidget(fig)


def plot_together(*figs: go.Figure) -> widgets.HBox:
    hbox = widgets.HBox([go.FigureWidget(fig) for fig in figs])
    return hbox


def plot_long_tail(table: pd.DataFrame, metric: str) -> go.Figure:
    plot_data = table.sort_values([metric], ascending=[False]).reset_index()
    fig = px.area(plot_data, x=plot_data.columns[0], y=metric)
    return fig


def plot_sample_recall_precision(n_recommendations=50) -> go.Figure:
    rec_range = [str(i + 1) for i in range(n_recommendations)]
    recommendations = pd.Series([random.random() < 0.5 for item in rec_range])
    instances = {}

    res1 = Results(pd.Series(rec_range), rec_range)
    res1.matches = recommendations.sort_values(ascending=False)
    instances['All at the start'] = res1

    res2 = Results(pd.Series(rec_range), rec_range)
    res2.matches = recommendations
    instances['Random'] = res2

    res3 = Results(pd.Series(rec_range), rec_range)
    res3.matches = recommendations.sort_values()
    instances['All at the end'] = res3

    return plot_recall_precision(instances)


def plot_recall_precision(recommendations: Results | dict[str, Results]) -> go.Figure:
    fig = go.Figure()
    if isinstance(recommendations, Results):
        recommendations = {'Model': recommendations}

    for name, results in recommendations.items():
        n_recs = results.n_recommendations
        rec_range = [str(i + 1) for i in range(n_recs)]
        recalls = [results.recall_at_k(k + 1) for k in range(n_recs)]
        precisions = [results.precision_at_k(k + 1) for k in range(n_recs)]
        average_precision = results.average_precision_at_k(n_recs)

        trace = go.Scatter(
            x=recalls,
            y=precisions,
            name=f'{name} - AP@{n_recs} = {average_precision:.2f}',
            text=rec_range,
            fill='tozeroy',
            mode='markers+lines',
            hovertemplate='k = %{text}<br><b>Recall:</b> %{x:.2f}<br><b>Precision:</b> %{y:.2f}<extra></extra>',
        )
        fig.add_trace(trace)
    fig.update_xaxes(title='Recall @ k')
    fig.update_yaxes(title='Precision @ k')
    fig.update_layout(title='Recall vs Precision plot')
    return fig
