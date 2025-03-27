# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 02:36:16 2025

@author: eulal
"""

import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==== Load raw data ====
raw_data = pd.read_csv("C:/Users/eulal/Documents/project 1 python/projeto 2 python/testData_2019_Civil - testData_2019_Civil.csv")
raw_data.rename(columns={"Civil (kWh)": "Power_kW", "Date": "Timestamp"}, inplace=True)
raw_data["Timestamp"] = pd.to_datetime(raw_data["Timestamp"])

# ==== Load forecast data ====
forecast_df = pd.read_csv("C:/Users/eulal/Documents/project 1 python/projeto 2 python/forecast_jan_mar_2019.csv")
forecast_df["Timestamp"] = pd.to_datetime(forecast_df["Timestamp"])

# ==== Forecast metrics ====
nmbe_static = 1.08
cvrmse_static = 8.23
r2_score = 0.01

# ==== Create dashboard ====
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("IST Civil Building - Forecast Dashboard", className="text-center mt-4 mb-4"),

    dcc.Tabs(id="tabs", value='tab1', children=[
        dcc.Tab(label='Raw Data (2019)', value='tab1'),
        dcc.Tab(label='Forecast vs Real (Jan–Mar)', value='tab2'),
        dcc.Tab(label='Data Exploration', value='tab3'),
        dcc.Tab(label='Feature Selection', value='tab4'),
        dcc.Tab(label='Regression & Metrics', value='tab5')
    ]),

    html.Div(id='tabs-content')
])

# ==== Callbacks ====

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab1':
        fig = px.line(raw_data, x='Timestamp', y='Power_kW', title="Energy Consumption (2019)")
        return html.Div([
            html.H4("Raw Data - Energy Consumption (kWh)"),
            dcc.Graph(figure=fig)
        ])

    elif tab == 'tab2':
        fig2 = px.line(forecast_df, x="Timestamp", y=["Power_kW", "Predicted_Power_kW"],
                       labels={"value": "Power (kW)", "variable": "Legend"},
                       title="Forecast vs Real (Jan–Mar 2019)")

        return html.Div([
            html.H4("Forecast vs Real - Jan to March 2019"),
            dcc.Graph(figure=fig2),
            html.H5("Model Metrics:"),
            html.Ul([
                html.Li(f"NMBE: {nmbe_static:.2f}%"),
                html.Li(f"cvRMSE: {cvrmse_static:.2f}%"),
                html.Li(f"R² Score: {r2_score:.2f}")
            ])
        ])

    elif tab == 'tab3':
        return html.Div([
            html.H4("Data Exploration"),
            html.Label("Select Variable to Visualize (Y):"),
            dcc.Dropdown(
                id='y-variable',
                options=[{'label': col, 'value': col} for col in raw_data.columns if col != 'Timestamp'],
                value='Power_kW'
            ),
            html.Label("Select Graph Type:"),
            dcc.Dropdown(
                id='graph-type',
                options=[
                    {'label': 'Line Chart', 'value': 'line'},
                    {'label': 'Scatter Plot', 'value': 'scatter'}
                ],
                value='line'
            ),
            dcc.Graph(id='exploration-graph')
        ])

    elif tab == 'tab4':
        return html.Div([
            html.H4("Feature Selection"),
            html.Label("Select Features to Analyze:"),
            dcc.Checklist(
                id='feature-list',
                options=[{'label': col, 'value': col} for col in raw_data.columns if col != 'Timestamp'],
                value=['Power_kW', 'temp_C'],
                inline=True
            ),
            html.Br(),
            html.Label("Select Feature Selection Method:"),
            dcc.Dropdown(
                id='selection-method',
                options=[
                    {'label': 'Correlation with Power_kW', 'value': 'correlation'},
                    {'label': 'Manual Selection', 'value': 'manual'}
                ],
                value='correlation'
            ),
            html.Br(),
            dcc.Graph(id='feature-selection-graph')
        ])

    elif tab == 'tab5':
        return html.Div([
            html.H4("Regression & Metrics"),
            html.Label("Choose Regression Model:"),
            dcc.Dropdown(
                id='regression-model',
                options=[
                    {'label': 'Random Forest', 'value': 'rf'},
                    {'label': 'Linear Regression', 'value': 'lr'}
                ],
                value='rf'
            ),
            html.Label("Choose Metrics to Display:"),
            dcc.Dropdown(
                id='metrics-selection',
                options=[
                    {'label': 'NMBE', 'value': 'nmbe'},
                    {'label': 'cvRMSE', 'value': 'cvrmse'},
                    {'label': 'Both', 'value': 'both'}
                ],
                value='both'
            ),
            html.Br(),
            html.Button("Run Model", id='run-model-btn', n_clicks=0),
            html.Br(), html.Br(),
            dcc.Graph(id='regression-plot'),
            html.Div(id='metrics-output', style={'marginTop': '20px'})
        ])

# ==== Extra Graph Callbacks ====

@app.callback(
    Output('exploration-graph', 'figure'),
    Input('y-variable', 'value'),
    Input('graph-type', 'value')
)
def update_exploration_graph(y_var, graph_type):
    if graph_type == 'line':
        fig = px.line(raw_data, x='Timestamp', y=y_var, title=f"{y_var} Over Time")
    else:
        fig = px.scatter(raw_data, x='Timestamp', y=y_var, title=f"{y_var} Over Time")
    return fig

@app.callback(
    Output('feature-selection-graph', 'figure'),
    Input('feature-list', 'value'),
    Input('selection-method', 'value')
)
def update_feature_graph(selected_features, method):
    if not selected_features:
        return px.bar(title="No features selected.")
    if method == 'correlation':
        corrs = raw_data[selected_features + ['Power_kW']].corr()['Power_kW'].drop('Power_kW')
        corrs = corrs.fillna(0)
        fig = px.bar(x=corrs.index.tolist(), y=corrs.values.tolist(),
                     labels={'x': 'Feature', 'y': 'Correlation'},
                     title="Correlation with Power_kW")
    else:
        fig = px.bar(x=selected_features, y=[1]*len(selected_features),
                     labels={'x': 'Feature', 'y': 'Selected'},
                     title="Manually Selected Features")
    return fig

# ==== Regression Callback ====

def nmbe(y_true, y_pred):
    return (np.mean(y_pred - y_true) / np.mean(y_true)) * 100

def cvrmse(y_true, y_pred):
    return (np.sqrt(np.mean((y_true - y_pred)**2)) / np.mean(y_true)) * 100

@app.callback(
    Output('regression-plot', 'figure'),
    Output('metrics-output', 'children'),
    Input('run-model-btn', 'n_clicks'),
    State('regression-model', 'value'),
    State('metrics-selection', 'value')
)
def run_model(n_clicks, selected_model, selected_metrics):
    if n_clicks == 0:
       return px.line(title="Run the model to see results."),

    features = ['temp_C', 'HR', 'windSpeed_m/s', 'solarRad_W/m2']
    df = raw_data.dropna(subset=features + ['Power_kW'])
    X = df[features]
    y = df['Power_kW']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if selected_model == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Real', 'y': 'Predicted'}, title="Real vs Predicted")
    fig.add_shape(type="line", x0=min(y_test), y0=min(y_test), x1=max(y_test), y1=max(y_test),
                  line=dict(color='green'))

    metrics_text = []
    if selected_metrics in ['nmbe', 'both']:
        metrics_text.append(f"NMBE: {nmbe(y_test, y_pred):.2f} %")
    if selected_metrics in ['cvrmse', 'both']:
        metrics_text.append(f"cvRMSE: {cvrmse(y_test, y_pred):.2f} %")

    metrics = " | ".join(metrics_text)
    return fig, metrics

# ==== Run App ====
if __name__ == '__main__':
    app.run_server(debug=True)

import webbrowser
webbrowser.open("http://127.0.0.1:8050")