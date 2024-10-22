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

# -----------------------------------------------------------------------------
# Global Variables
# -----------------------------------------------------------------------------
trained_models = {}
feature_cols = []

# -----------------------------------------------------------------------------
# Load the Default Dataset
# -----------------------------------------------------------------------------
def load_and_preprocess_data():
    # Load default data
    raw_data = load_default_data()
    # Preprocess data
    preprocessed_data, preprocessing_metrics, numerical_columns, categorical_columns, additional_numerical_features, numerical_means, numerical_stds = preprocess_data(raw_data)
    
    # Serialize both raw and preprocessed data
    raw_data_json = raw_data.to_json(date_format='iso', orient='split')
    data_json = preprocessed_data.to_json(date_format='iso', orient='split')
    
    # Store both in `store_data`
    store_data = {
        'raw_data': raw_data_json,  # Include raw_data here
        'data': data_json,
        'metrics': preprocessing_metrics,
        'numerical_means': numerical_means,
        'numerical_stds': numerical_stds,
        'numerical_columns': numerical_columns,
        'categorical_columns': categorical_columns,
        'additional_numerical_features': additional_numerical_features
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
    # Retrieve necessary columns
    numerical_columns = store_data['numerical_columns']
    additional_numerical_features = store_data['additional_numerical_features']
    # Train models
    model_performance, trained_models, feature_cols = train_models(data, numerical_columns, additional_numerical_features)
    store_data['model_performance'] = model_performance
    # Store model names instead of models themselves
    store_data['trained_models_names'] = {model_name: trained_models[model_name].__class__.__name__ for model_name in trained_models}
    # Serialize model_performance for storage
    store_data['model_performance_json'] = json.dumps(model_performance)
    store_data['feature_cols'] = feature_cols
    logger.info(f"Feature columns used for training: {feature_cols}")
    return store_data

store_data = train_and_store_models(store_data)

# -----------------------------------------------------------------------------
# Fairness Analysis
# -----------------------------------------------------------------------------
def perform_fairness_analysis(store_data):
    global trained_models
    data = pd.read_json(store_data['data'], orient='split')
    fairness_metrics = fairness_analysis(data, trained_models, store_data['feature_cols'])
    store_data['fairness_metrics'] = fairness_metrics
    store_data['fairness_metrics_json'] = json.dumps(fairness_metrics)
    return store_data

store_data = perform_fairness_analysis(store_data)
logger.critical("Dash is running on http://127.0.0.1:8050/.")
# -----------------------------------------------------------------------------
# Set the Layout with Store Data
# -----------------------------------------------------------------------------
app.layout = serve_layout(store_data)

# -----------------------------------------------------------------------------
# Register Callbacks
# -----------------------------------------------------------------------------
register_render_content_callbacks(app, trained_models)
register_prediction_callbacks(app, trained_models)
register_import_data_callbacks(app, trained_models)

# -----------------------------------------------------------------------------
# Run the Dash App
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
