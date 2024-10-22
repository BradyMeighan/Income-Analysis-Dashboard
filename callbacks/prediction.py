# callbacks/prediction.py

# =============================================================================
# Prediction Callback
# =============================================================================

from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
from dash import html
import dash
import pandas as pd
from data.preprocessing import preprocess_data
import json
import logging
import numpy as np

logger = logging.getLogger()

def register_prediction_callbacks(app, trained_models):
    @app.callback(
        [Output('prediction-output', 'children'),
         Output('prediction-history', 'data')],
        Input('predict-button', 'n_clicks'),
        State('model-dropdown', 'value'),
        State({'type': 'input', 'index': ALL}, 'value'),
        State({'type': 'input', 'index': ALL}, 'id'),
        State('stored-data', 'data'),
        State('prediction-history', 'data'),
        prevent_initial_call=True
    )
    def handle_prediction_and_history(n_clicks, model_name, input_values, input_ids, store_data, prediction_history):
        # Retrieve necessary variables from store_data
        numerical_columns = store_data.get('numerical_columns', [])
        additional_numerical_features = store_data.get('additional_numerical_features', [])
        numerical_means = store_data.get('numerical_means', {})
        numerical_stds = store_data.get('numerical_stds', {})
        feature_cols = store_data.get('feature_cols', [])

        # Use trained_models passed as argument
        model = trained_models.get(model_name, None)
        if model is None:
            error_message = "Selected model is not available."
            logger.error(error_message)
            return dbc.Alert(error_message, color="danger"), prediction_history
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate
    
        # Map input values to feature names
        input_dict = {}
        for value, id_dict in zip(input_values, input_ids):
            feature_name = id_dict['index']
            input_dict[feature_name] = value
    
        # Create a DataFrame from input values
        input_df = pd.DataFrame([input_dict])
    
        # Retrieve numerical_means and numerical_stds from store_data
        numerical_means = store_data.get('numerical_means', {})
        numerical_stds = store_data.get('numerical_stds', {})
    
        if not numerical_means or not numerical_stds:
            error_message = "Numerical means and standard deviations are missing from stored data."
            logger.error(error_message)
            return dbc.Alert(error_message, color="danger"), prediction_history
    
        # Preprocess the input data
        preprocessed_input_data, _, _, _, _, _, _ = preprocess_data(input_df, is_prediction=True, numerical_means=numerical_means, numerical_stds=numerical_stds)
    
        # Encode categorical variables
        X_input = preprocessed_input_data.drop(columns=['income_numeric'], errors='ignore')
        X_input = pd.get_dummies(X_input, drop_first=True)
        logger.info("Encoded categorical variables for input data.")
    
        # Ensure the input data has the same dummy columns as the training data
        X_train = pd.read_json(store_data['data'], orient='split')[feature_cols]
        X_train = pd.get_dummies(X_train, drop_first=True)
        X_input = X_input.reindex(columns=X_train.columns, fill_value=0)
        logger.info("Reindexed input data to match training dummy variables.")
    
        # Retrieve the selected model
        #trained_models = {model: eval(model) for model in store_data.get('trained_models', {})}
        model = trained_models.get(model_name, None)
        if model is None:
            error_message = "Selected model is not available."
            logger.error(error_message)
            return dbc.Alert(error_message, color="danger"), prediction_history
    
        # Make prediction
        try:
            if model_name == 'XGBoost':
                X_input_mod = X_input.values.astype(np.float32)
                prediction = model.predict(X_input_mod)
            else:
                prediction = model.predict(X_input)
        except Exception as e:
            error_message = f"An error occurred during prediction: {e}"
            logger.error(error_message)
            return dbc.Alert(error_message, color="danger"), prediction_history
    
        # Map prediction to income category
        income_map = {0: '<=50K', 1: '>50K'}
        predicted_income = income_map.get(int(prediction[0]), "Unknown")
    
        # Store the input values and prediction in history
        input_dict['Predicted Income'] = predicted_income
        input_dict['Model'] = model_name
        prediction_history.append(input_dict)
    
        # Limit history to the last 10 predictions
        prediction_history = prediction_history[-10:]
    
        # Display the result
        result = dbc.Alert(
            f"The predicted income category is: {predicted_income}",
            color="success",
            style={'font-size': '20px'}
        )
    
        return result, prediction_history
