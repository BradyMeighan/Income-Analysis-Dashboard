# =============================================================================
# Income Analysis Dashboard - Main Application
# =============================================================================

import pandas as pd
import json
from dash import Dash
import dash_bootstrap_components as dbc
from layout.layout import serve_layout
from data.preprocessing import load_default_data, preprocess_data
from models.training import train_models
from fairness.analysis import fairness_analysis
from callbacks.render_content import register_render_content_callbacks
from callbacks.prediction import register_prediction_callbacks
from callbacks.import_data import register_import_data_callbacks
from utils.logging_config import logger

# Import necessary Dash components for callbacks
from dash.dependencies import Input, Output, State

# -----------------------------------------------------------------------------
# Initialize Dash Application
# -----------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "Income Analysis Dashboard"
app.layout = serve_layout()

# -----------------------------------------------------------------------------
# Global Variables
# -----------------------------------------------------------------------------
trained_models = {}
preprocessed_data = None
numerical_columns = []
additional_numerical_features = []
categorical_columns = []
numerical_means = {}
numerical_stds = {}
feature_cols = []

# -----------------------------------------------------------------------------
# Load the Default Dataset
# -----------------------------------------------------------------------------
def load_and_preprocess_data():
    global raw_data, preprocessed_data, preprocessing_metrics, numerical_columns, categorical_columns, additional_numerical_features, numerical_means, numerical_stds, feature_cols
    raw_data = load_default_data()
    preprocessed_data, preprocessing_metrics, numerical_columns, categorical_columns, additional_numerical_features, numerical_means, numerical_stds = preprocess_data(raw_data)
    data_json = preprocessed_data.to_json(date_format='iso', orient='split')
    store_data = {
        'data': data_json,
        'metrics': preprocessing_metrics,
        'numerical_means': numerical_means,
        'numerical_stds': numerical_stds
    }
    return store_data

store_data = load_and_preprocess_data()

# -----------------------------------------------------------------------------
# Train Models
# -----------------------------------------------------------------------------
def train_and_store_models(store_data):
    global trained_models, feature_cols
    # Load preprocessed data
    data = pd.read_json(store_data['data'], orient='split')
    # Train models
    model_performance, trained_models, feature_cols = train_models(data, numerical_columns, additional_numerical_features)
    store_data['model_performance'] = model_performance
    store_data['trained_models'] = {model: trained_models[model].__class__.__name__ for model in trained_models}
    
    # Serialize model_performance for storage
    store_data['model_performance_json'] = json.dumps(model_performance)
    logger.info(f"Feature columns used for training: {feature_cols}")
    return store_data

store_data = train_and_store_models(store_data)

# -----------------------------------------------------------------------------
# Fairness Analysis
# -----------------------------------------------------------------------------
def perform_fairness_analysis(store_data):
    global fairness_metrics
    data = pd.read_json(store_data['data'], orient='split')
    fairness_metrics = fairness_analysis(data, trained_models, feature_cols)
    store_data['fairness_metrics'] = fairness_metrics
    store_data['fairness_metrics_json'] = json.dumps(fairness_metrics)
    return store_data

store_data = perform_fairness_analysis(store_data)

# -----------------------------------------------------------------------------
# Update Stored Data (Client-side Callback)
# -----------------------------------------------------------------------------
app.clientside_callback(
    """
    function(storeData) {
        return storeData;
    }
    """,
    Output('stored-data', 'data'),
    Input('stored-data', 'data'),
    prevent_initial_call=True
)

app.layout = serve_layout()

# -----------------------------------------------------------------------------
# Server-side Callback to Update Stored Data
# -----------------------------------------------------------------------------
@app.callback(
    Output('stored-data', 'data'),
    Input('stored-data', 'data'),
    prevent_initial_call=True
)
def update_store_data(store_data_input):
    return store_data

# -----------------------------------------------------------------------------
# Register Callbacks
# -----------------------------------------------------------------------------
register_render_content_callbacks(app)
register_prediction_callbacks(app)
register_import_data_callbacks(app)

# -----------------------------------------------------------------------------
# Run the Dash App
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
